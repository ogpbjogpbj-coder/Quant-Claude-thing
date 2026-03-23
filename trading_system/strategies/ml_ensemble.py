"""ML Ensemble strategy using XGBoost + LightGBM.

Trains on historical features to predict forward returns. Combines
multiple models with stacking for robust signal generation.
"""

import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

from trading_system.config import StrategyMLEnsembleConfig
from trading_system.strategies.base import BaseStrategy, Signal


MODEL_DIR = Path(__file__).parent.parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class MLEnsembleStrategy(BaseStrategy):
    name = "ml_ensemble"

    def __init__(self, config: StrategyMLEnsembleConfig):
        self.config = config
        self.features = config.features
        self.models: dict = {}
        self.scalers: dict = {}
        self.last_train_time: Optional[datetime] = None
        self._load_models()

    def _load_models(self) -> None:
        """Load pre-trained models from disk if available."""
        model_path = MODEL_DIR / "ml_ensemble.pkl"
        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    saved = pickle.load(f)
                self.models = saved.get("models", {})
                self.scalers = saved.get("scalers", {})
                self.last_train_time = saved.get("train_time")
                logger.info(f"Loaded ML models trained at {self.last_train_time}")
            except Exception as e:
                logger.warning(f"Failed to load ML models: {e}")

    def _save_models(self) -> None:
        """Persist trained models to disk."""
        try:
            with open(MODEL_DIR / "ml_ensemble.pkl", "wb") as f:
                pickle.dump({
                    "models": self.models,
                    "scalers": self.scalers,
                    "train_time": self.last_train_time,
                }, f)
            logger.info("ML models saved to disk")
        except Exception as e:
            logger.error(f"Failed to save ML models: {e}")

    def _needs_retrain(self) -> bool:
        if self.last_train_time is None:
            return True
        hours_since = (datetime.utcnow() - self.last_train_time).total_seconds() / 3600
        return hours_since >= self.config.retrain_interval_hours

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Extract feature matrix from indicator-enriched DataFrame."""
        available = [f for f in self.features if f in df.columns]
        if len(available) < len(self.features) * 0.7:
            return None

        feat_df = df[available].copy()
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
        feat_df = feat_df.ffill().bfill()

        if feat_df.isna().sum().sum() > 0:
            feat_df = feat_df.fillna(0)

        return feat_df

    def _prepare_target(self, df: pd.DataFrame, forward_days: int = 5) -> pd.Series:
        """Target: forward N-day return, classified as 0 (sell), 1 (hold), 2 (buy)."""
        fwd_ret = df["close"].pct_change(forward_days).shift(-forward_days)
        # Classify: >1% = buy(2), <-1% = sell(0), else hold(1)
        target = pd.Series(1, index=fwd_ret.index)
        target[fwd_ret > 0.01] = 2
        target[fwd_ret < -0.01] = 0
        return target

    def train(self, data: dict[str, pd.DataFrame]) -> None:
        """Train the ML ensemble on historical data."""
        try:
            import xgboost as xgb
            import lightgbm as lgb
        except ImportError:
            logger.warning("XGBoost/LightGBM not installed, skipping ML training")
            return

        logger.info("Training ML ensemble models...")

        all_X = []
        all_y = []

        for sym, df in data.items():
            if len(df) < 60:
                continue

            features = self._prepare_features(df)
            if features is None:
                continue

            target = self._prepare_target(df)
            # Remove last 5 rows (no target) and align
            valid_mask = target.notna()
            X = features[valid_mask]
            y = target[valid_mask]

            if len(X) > 20:
                all_X.append(X)
                all_y.append(y)

        if not all_X:
            logger.warning("Insufficient data for ML training")
            return

        X_train = pd.concat(all_X)
        y_train = pd.concat(all_y)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        self.scalers["main"] = scaler

        # TimeSeriesSplit for validation
        tscv = TimeSeriesSplit(n_splits=3)
        scores = []

        # Train XGBoost
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            verbosity=0,
            random_state=42,
        )

        for train_idx, val_idx in tscv.split(X_scaled):
            xgb_model.fit(
                X_scaled[train_idx], y_train.iloc[train_idx],
                eval_set=[(X_scaled[val_idx], y_train.iloc[val_idx])],
                verbose=False,
            )
            score = xgb_model.score(X_scaled[val_idx], y_train.iloc[val_idx])
            scores.append(score)

        # Final fit on all data
        xgb_model.fit(X_scaled, y_train, verbose=False)
        self.models["xgboost"] = xgb_model

        # Train LightGBM
        lgb_model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multiclass",
            num_class=3,
            verbosity=-1,
            random_state=42,
        )
        lgb_model.fit(X_scaled, y_train)
        self.models["lightgbm"] = lgb_model

        self.last_train_time = datetime.utcnow()
        avg_score = np.mean(scores) if scores else 0
        logger.info(f"ML training complete. XGB CV accuracy: {avg_score:.3f}")

        self._save_models()

    def generate_signals(
        self,
        data: dict[str, pd.DataFrame],
        current_positions: dict[str, float],
    ) -> list[Signal]:
        # Retrain if needed
        if self._needs_retrain():
            self.train(data)

        if not self.models or "main" not in self.scalers:
            logger.debug("ML models not trained yet, skipping")
            return []

        signals = []
        scaler = self.scalers["main"]

        for sym, df in data.items():
            if df.empty or len(df) < 30:
                continue

            try:
                features = self._prepare_features(df)
                if features is None:
                    continue

                # Get latest row
                latest = features.iloc[[-1]]
                X = scaler.transform(latest)

                # Ensemble prediction: average probabilities
                probs_list = []
                for model_name, model in self.models.items():
                    try:
                        probs = model.predict_proba(X)[0]
                        probs_list.append(probs)
                    except Exception as e:
                        logger.debug(f"Model {model_name} prediction failed: {e}")

                if not probs_list:
                    continue

                # Average probabilities across models
                avg_probs = np.mean(probs_list, axis=0)

                # Classes: [0=sell, 1=hold, 2=buy]
                sell_prob = avg_probs[0]
                hold_prob = avg_probs[1]
                buy_prob = avg_probs[2]

                # Generate signal based on probability differential
                direction = buy_prob - sell_prob  # -1 to +1
                confidence = max(buy_prob, sell_prob)  # How sure the model is

                # Only signal if model is reasonably confident
                if confidence < 0.4 or abs(direction) < 0.15:
                    continue

                atr = df.iloc[-1].get("atr_14", 0)
                close = df.iloc[-1]["close"]

                signal_kwargs = {
                    "symbol": sym,
                    "direction": np.clip(direction, -1.0, 1.0),
                    "confidence": np.clip(confidence, 0.0, 0.95),
                    "strategy": self.name,
                    "metadata": {
                        "buy_prob": float(buy_prob),
                        "sell_prob": float(sell_prob),
                        "hold_prob": float(hold_prob),
                    },
                }

                if direction > 0 and atr > 0:
                    signal_kwargs["stop_loss"] = close - 2 * atr
                    signal_kwargs["take_profit"] = close + 3.5 * atr

                signals.append(Signal(**signal_kwargs))

            except Exception as e:
                logger.debug(f"ML signal generation failed for {sym}: {e}")

        return signals
