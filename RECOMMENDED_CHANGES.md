# Recommended Changes for QuantClaude

A prioritized list of bugs, improvements, and missing capabilities identified through a full codebase review.

---

## Critical Bugs (Fix Before Any Live Trading)

### 1. `_recently_closed` dict is never populated — exit journaling broken

**Files:** `trading_system/orchestrator.py:381`, `trading_system/portfolio.py`

The `_recently_closed` property (orchestrator.py:520-524) initializes an empty dict, but nothing ever writes to it. `_journal_closed_positions()` iterates over it (line 381) expecting entries like `{symbol: {"reason": ..., "price": ..., "tracked": ...}}`, but they never appear.

**Impact:** Exit trades are never recorded in the trade journal, which means the adaptive learning system (`strategy_evolver.py`) has no completed trade data to learn from. The entire feedback loop is broken.

**Fix:** In `portfolio.py`, when a sell order closes a position (lines 327-350), populate `orchestrator._recently_closed[sym]` with exit metadata before removing it from `tracked`. Alternatively, have `check_stops()` return richer exit info and let the orchestrator populate it in `_check_stops_with_journal()`.

---

### 2. `est_size` undefined in short-selling branch — NameError crash

**File:** `trading_system/portfolio.py:354`

The short-selling branch (line 352-354) references `est_size`, which is only defined in the long-entry branch (line 238). If a short-sell signal arrives for a symbol with no existing position, this crashes with `NameError`.

**Fix:** Add before line 354:
```python
est_size = self._equity * self.config.risk.max_position_size_pct / 100 * 0.5
```

---

### 3. Signal metadata stores raw DataFrames — serialization failures

**File:** `trading_system/orchestrator.py:297`

Enriched market data (pandas DataFrames) is stored directly into signal metadata via `enriched.get(sig.symbol)`. When the trade journal attempts to serialize this to JSON for persistence, it will fail or produce garbage.

**Fix:** Either exclude the DataFrame from metadata before journaling, or store only summary statistics (e.g., last close, volume, indicator values).

---

## High-Severity Issues

### 4. Hardcoded correlation hard-block at 0.85

**File:** `trading_system/portfolio.py:183`

If any existing position has correlation > 0.85 with a new signal, the position is completely blocked (`return 0.0`). The configurable `max_correlation_threshold` (line 182) is only used for soft scaling. The 0.85 hard limit is not configurable.

**Recommendation:** Make the hard-block threshold configurable, or remove it entirely and rely on the soft-scaling logic which already handles high correlation gracefully.

---

### 5. No API reconnection logic

**Files:** `trading_system/execution.py`, `trading_system/data_ingestion.py`

The Alpaca `TradingClient` is created once during `__init__` and never recreated. If the connection drops (network blip, token refresh), the system silently stops executing trades.

**Recommendation:** Add a health-check or try/except wrapper around API calls that recreates the client on connection errors, with exponential backoff.

---

### 6. Sentiment strategy has no rate-limit handling

**File:** `trading_system/strategies/sentiment.py`

The Alpaca news API is called without rate-limit detection. If a 429 response is returned, the strategy silently fails.

**Recommendation:** Add retry logic with backoff for 429 responses, or add a request-rate limiter.

---

### 7. TWAP background thread has no timeout

**File:** `trading_system/execution.py:222`

The TWAP slicing loop runs in a background thread with no maximum duration. If a slice fails repeatedly or the loop stalls, the thread leaks.

**Recommendation:** Add a maximum execution time (e.g., 30 minutes) after which the TWAP thread cancels remaining slices and logs a warning.

---

## Medium-Severity Issues

### 8. Bare `except` clauses swallow errors

**Files:** `strategy_evolver.py:631,643`, `orchestrator.py:605,613,621`, `backtester.py:174`, `trade_journal.py:866`

Multiple bare `except:` or broad `except Exception` blocks catch and silently discard errors. This masks bugs during development and production.

**Recommendation:** At minimum, log the exception with `logger.exception()` in every catch block. Replace bare `except:` with `except Exception:`.

---

### 9. Database queries lack proper indexes

**File:** `trading_system/utils/db.py`

Queries filter by `symbol` and `timestamp` columns but the schema only has primary key indexes. As the trade journal grows past ~10K records, queries slow significantly.

**Recommendation:** Add composite indexes:
```sql
CREATE INDEX IF NOT EXISTS idx_trades_symbol_ts ON trades(symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_ts ON snapshots(timestamp);
```

---

### 10. Hardcoded sector map covers only ~30 symbols

**File:** `trading_system/portfolio.py:20-45`

The sector map is a static dict of ~30 symbols. The dynamic universe scanner expands the symbol pool to ~500, all of which fall under the default "Other" sector. Sector risk limits become meaningless.

**Recommendation:** Fetch sector data from Alpaca asset metadata or a free API (e.g., Financial Modeling Prep), and cache it. Fall back to the static map only when unavailable.

---

### 11. Test coverage is minimal (<5%)

**File:** `tests/test_strategies.py`

Only strategy signal generation is tested (273 lines). No tests exist for:
- Risk manager circuit breakers
- Portfolio position sizing and stop-loss logic
- Order execution and partial fills
- Trade journal recording
- Signal aggregation
- Database operations

**Recommendation:** Prioritize integration tests for the trading loop (`orchestrator.py`) and unit tests for `portfolio.py` and `risk_manager.py`, since bugs there directly lose money.

---

## Low-Severity / Improvements

### 12. No graceful shutdown in `main.py`

`main.py` doesn't handle SIGTERM/SIGINT. The orchestrator has shutdown logic, but it's not wired to the CLI entry point.

### 13. No health checks in Docker Compose

The `docker-compose.yaml` services have no health checks, so Docker can't auto-restart a hung trading bot.

### 14. No database backup mechanism

Trade history lives in a single SQLite file with no backup or replication. A disk failure loses all historical data.

### 15. ML model staleness not detected

`ml_ensemble.py` retrains every 24 hours, but if training fails repeatedly, stale models are used indefinitely with no warning.

### 16. No README or architecture documentation

There is no README.md explaining setup, configuration, strategy descriptions, or system architecture. This makes onboarding difficult and increases the risk of misconfiguration.

---

## Summary

| Priority | Count | Action |
|----------|-------|--------|
| Critical bugs | 3 | Fix immediately — blocks core functionality |
| High severity | 4 | Fix before paper trading validation |
| Medium severity | 4 | Fix before production |
| Low / improvements | 5 | Address as time permits |

The three critical bugs (#1, #2, #3) should be fixed first, as they break the adaptive learning feedback loop and crash short-selling. After those, adding test coverage (#11) will prevent regressions as other issues are addressed.
