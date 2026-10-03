# QQQ 15-Minute Opening Range Breakout — Strategy Specification

## 1. Research Question

Does a 15-minute Opening Range Breakout (ORB) strategy with 5-minute
breakout confirmation exhibit a persistent trading edge on QQQ?

The longer-term objective of this project is to investigate whether the
strategy's return distribution is suitable for proprietary trading firm
challenge environments.

---

## 2. Instrument

- Asset: QQQ
- Market: NASDAQ
- Data frequency: 5-minute OHLCV
- Timezone: America/New_York
- Session: Regular Trading Hours (RTH)
- Pre-market and after-hours data are excluded.

---

## 3. Research Period

### Development / In-Sample

January 1, 2017 – December 31, 2022

This period will be used for implementation, debugging, and development
of the backtesting framework.

### Out-of-Sample

January 1, 2023 – September 30, 2026

The out-of-sample period is reserved as a holdout dataset.

The strategy specification will be frozen before evaluating the
out-of-sample period. Strategy parameters will not be modified based
on out-of-sample results.

---

## 4. Opening Range

The opening range is defined using the first 15 minutes of the regular
trading session:

09:30 – 09:45 ET

Using 5-minute bars, the opening range consists of:

- 09:30 – 09:35
- 09:35 – 09:40
- 09:40 – 09:45

Opening Range High (ORH):

ORH = highest High of the three opening-range candles.

Opening Range Low (ORL):

ORL = lowest Low of the three opening-range candles.

The opening range becomes fixed at 09:45 ET.

---

## 5. Entry Window

Breakout signals are monitored from 09:45 ET until 12:00 ET.

Only the first valid breakout signal of the day is traded.

Maximum trades per day: 1.

### Long Signal

A long signal occurs when a 5-minute candle closes strictly above
the Opening Range High.

Signal condition:

Close > ORH

The position is entered at the OPEN of the following 5-minute candle.

### Short Signal

A short signal occurs when a 5-minute candle closes strictly below
the Opening Range Low.

Signal condition:

Close < ORL

The position is entered at the OPEN of the following 5-minute candle.

### No Breakout

If neither a valid long nor short breakout occurs during the entry
window, no trade is taken that day.

No new signals are accepted after 12:00 ET.

---

## 6. Stop Loss

The opposite side of the opening range is used as the initial stop.

### Long Position

Stop = ORL

### Short Position

Stop = ORH

The stop remains fixed throughout the trade.

No trailing stop is used.

---

## 7. Initial Risk

Risk per share is determined at the time of entry.

### Long

R = Entry Price - ORL

### Short

R = ORH - Entry Price

Trades with invalid or non-positive risk values are rejected.

Initial strategy evaluation will be performed primarily in R-multiples
rather than using a fixed account size.

---

## 8. Profit Target

The strategy uses a fixed 1:2 risk-reward ratio.

### Long

Target = Entry + 2R

### Short

Target = Entry - 2R

Therefore:

- Stop Loss = -1R
- Take Profit = +2R

---

## 9. Trade Management

Once a position has been opened:

- Stop loss remains fixed.
- Profit target remains fixed.
- No partial profit taking is used.
- No trailing stop is used.
- No additional positions are opened.
- No reversal trades are allowed.

---

## 10. Time Exit

Any position that remains open at 15:00 ET is closed.

The exit price for the time-based exit will be defined using the
available 5-minute bar data in the backtesting implementation.

No positions are carried overnight.

---

## 11. Daily Trading Rules

- Maximum one trade per trading day.
- First valid breakout only.
- Opening range: 09:30–09:45 ET.
- Breakout confirmation timeframe: 5 minutes.
- Entry window: 09:45–12:00 ET.
- Entry occurs at the next 5-minute candle open after confirmation.
- No re-entry after a stopped-out trade.
- No reversal after entry.
- Fixed opposite-side opening-range stop.
- Fixed 2R profit target.
- All remaining positions are closed at 15:00 ET.

---

## 12. Position Sizing

The baseline strategy will initially be evaluated using R-multiples.

This separates the underlying trading edge from account-specific
position-sizing assumptions.

Account-level position sizing will be introduced during the later
prop-firm analysis.

Examples of risk-per-trade levels may subsequently include:

- 0.25%
- 0.50%
- 0.75%
- 1.00%

These values are NOT part of the baseline strategy specification.

---

## 13. Transaction Costs and Execution

The initial baseline backtest will first evaluate gross strategy
performance.

Execution occurs at the next 5-minute candle open following a valid
breakout signal.

Subsequent robustness tests will incorporate:

- Commissions
- Bid-ask spread
- Slippage

Results with realistic transaction costs will be reported separately
from gross results.

---

## 14. Baseline Performance Metrics

The baseline backtest will report at minimum:

### Trade Statistics

- Number of trades
- Number of long trades
- Number of short trades
- Win rate
- Average R per trade
- Median R per trade
- Profit factor
- Average winner
- Average loser
- Maximum consecutive wins
- Maximum consecutive losses

### Portfolio / Strategy Statistics

- Cumulative R
- Maximum drawdown in R
- Sharpe ratio
- Sortino ratio
- Monthly performance
- Annual performance

### Diagnostic Analysis

- Long vs. short performance
- Distribution of trade returns
- Holding-time distribution
- Performance by year

---

## 15. Research Integrity

The baseline strategy rules are defined before examining the
out-of-sample results.

The initial strategy parameters are:

- 15-minute opening range
- 5-minute breakout confirmation
- next-bar-open execution
- opposite-side opening-range stop
- 2R profit target
- one trade per day
- 12:00 ET entry cutoff
- 15:00 ET forced exit

These parameters should not be modified in response to out-of-sample
performance.

Any alternative parameters tested later will be explicitly labeled as
robustness or sensitivity analysis rather than part of the original
baseline strategy.