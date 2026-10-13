from pathlib import Path
from datetime import time, datetime, timedelta

import numpy as np
import pandas as pd

from src.orb_strategy import (
    load_development_data,
    simulate_trade,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "parameter_sweep_results.csv"
)


# ============================================================
# PARAMETER GRID
# ============================================================

OR_MINUTES = [
    5,
    10,
    15,
    30,
    60,
]

TARGET_RS = [
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
]

SIGNAL_CUTOFFS = [
    time(10, 30),
    time(11, 0),
    time(12, 0),
    time(13, 0),
]

# ============================================================
# TIME HELPERS
# ============================================================

def add_minutes(base_time, minutes):
    """
    Add minutes to a datetime.time object.
    """

    base = datetime.combine(
        datetime.today(),
        base_time,
    )

    result = base + timedelta(minutes=minutes)

    return result.time()


# ============================================================
# PARAMETERIZED OPENING RANGE
# ============================================================

def get_opening_range(
    day,
    or_minutes,
):
    """
    Opening range begins at 09:30 and lasts
    for or_minutes.

    Examples:

    5m:
        09:30 bar

    15m:
        09:30, 09:35, 09:40

    30m:
        09:30 ... 09:55

    60m:
        09:30 ... 10:25
    """

    or_start = time(9, 30)

    or_end = add_minutes(
        or_start,
        or_minutes,
    )

    opening_range = day[
        (day["timestamp"].dt.time >= or_start)
        & (day["timestamp"].dt.time < or_end)
    ]

    expected_bars = or_minutes // 5

    if len(opening_range) != expected_bars:
        return None

    return {
        "or_high": opening_range["high"].max(),
        "or_low": opening_range["low"].min(),
        "or_end": or_end,
    }


# ============================================================
# PARAMETERIZED BREAKOUT
# ============================================================

def find_breakout(
    day,
    or_high,
    or_low,
    signal_start,
    signal_end,
):
    """
    Find first valid close outside opening range.
    """

    signal_bars = day[
        (day["timestamp"].dt.time >= signal_start)
        & (day["timestamp"].dt.time <= signal_end)
    ]

    for idx, bar in signal_bars.iterrows():

        if bar["close"] > or_high:
            return idx, "long"

        if bar["close"] < or_low:
            return idx, "short"

    return None


# ============================================================
# ENTRY
# ============================================================

def get_entry(
    day,
    signal_idx,
):
    """
    Entry occurs at the open of the next
    5-minute bar.
    """

    day_indices = list(day.index)

    try:
        signal_position = day_indices.index(
            signal_idx
        )
    except ValueError:
        return None

    entry_position = signal_position + 1

    if entry_position >= len(day_indices):
        return None

    entry_idx = day_indices[entry_position]
    entry_bar = day.loc[entry_idx]

    return entry_idx, entry_bar


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_day_parameterized(
    day,
    or_minutes,
    target_r,
    signal_cutoff,
):
    """
    Run one session using the supplied parameter
    combination.

    The original orb_strategy.py is not modified.
    """

    day = day.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # Opening range
    # --------------------------------------------------------

    opening_range = get_opening_range(
        day,
        or_minutes,
    )

    if opening_range is None:
        return None

    or_high = opening_range["or_high"]
    or_low = opening_range["or_low"]
    signal_start = opening_range["or_end"]

    # If OR itself finishes after cutoff,
    # this configuration cannot generate a trade.
    if signal_start > signal_cutoff:
        return None

    # --------------------------------------------------------
    # Breakout
    # --------------------------------------------------------

    breakout = find_breakout(
        day=day,
        or_high=or_high,
        or_low=or_low,
        signal_start=signal_start,
        signal_end=signal_cutoff,
    )

    if breakout is None:
        return None

    signal_idx, direction = breakout

    signal_bar = day.loc[signal_idx]

    # --------------------------------------------------------
    # Entry
    # --------------------------------------------------------

    entry = get_entry(
        day,
        signal_idx,
    )

    if entry is None:
        return None

    entry_idx, entry_bar = entry

    entry_price = entry_bar["open"]

    # --------------------------------------------------------
    # Stop / risk / target
    # --------------------------------------------------------

    if direction == "long":

        stop_price = or_low

        initial_risk = (
            entry_price - stop_price
        )

        if initial_risk <= 0:
            return None

        target_price = (
            entry_price
            + target_r * initial_risk
        )

    elif direction == "short":

        stop_price = or_high

        initial_risk = (
            stop_price - entry_price
        )

        if initial_risk <= 0:
            return None

        target_price = (
            entry_price
            - target_r * initial_risk
        )

    else:
        raise ValueError(
            f"Unknown direction: {direction}"
        )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Use the ORIGINAL trade simulation engine.
    # We do not modify its execution behaviour here.
    # --------------------------------------------------------

    trade_result = simulate_trade(
        day=day,
        entry_idx=entry_idx,
        direction=direction,
        stop_price=stop_price,
        target_price=target_price,
    )

    if trade_result is None:
        return None

    exit_idx = trade_result["exit_idx"]
    exit_bar = day.loc[exit_idx]
    exit_price = trade_result["exit_price"]

    # --------------------------------------------------------
    # PnL
    # --------------------------------------------------------

    if direction == "long":

        pnl = (
            exit_price - entry_price
        )

    else:

        pnl = (
            entry_price - exit_price
        )

    r_multiple = (
        pnl / initial_risk
    )

    return {
        "date": day["date"].iloc[0],
        "direction": direction,
        "signal_time": signal_bar["timestamp"],
        "entry_time": entry_bar["timestamp"],
        "entry_price": entry_price,
        "exit_time": exit_bar["timestamp"],
        "exit_price": exit_price,
        "exit_reason": trade_result["exit_reason"],
        "initial_risk": initial_risk,
        "r_multiple": r_multiple,
    }


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(trades):
    """
    Calculate performance metrics in R-space.
    """

    if len(trades) == 0:
        return None

    r = trades["r_multiple"]

    winners = r[r > 0]
    losers = r[r < 0]

    # --------------------------------------------------------
    # Profit factor
    # --------------------------------------------------------

    gross_profit = winners.sum()

    gross_loss = abs(
        losers.sum()
    )

    if gross_loss > 0:
        profit_factor = (
            gross_profit / gross_loss
        )
    else:
        profit_factor = np.nan

    # --------------------------------------------------------
    # R equity curve / drawdown
    # --------------------------------------------------------

    equity = r.cumsum()

    running_max = equity.cummax()

    drawdown = (
        equity - running_max
    )

    max_drawdown_r = abs(
        drawdown.min()
    )

    # --------------------------------------------------------
    # Trade-level Sharpe-like statistic
    #
    # Not annualized yet.
    # --------------------------------------------------------

    if r.std(ddof=1) > 0:

        trade_sharpe = (
            r.mean()
            / r.std(ddof=1)
        )

    else:
        trade_sharpe = np.nan

    # --------------------------------------------------------
    # Long / short
    # --------------------------------------------------------

    longs = trades[
        trades["direction"] == "long"
    ]

    shorts = trades[
        trades["direction"] == "short"
    ]

    long_avg_r = (
        longs["r_multiple"].mean()
        if len(longs) > 0
        else np.nan
    )

    short_avg_r = (
        shorts["r_multiple"].mean()
        if len(shorts) > 0
        else np.nan
    )

    # --------------------------------------------------------
    # Losing streak
    # --------------------------------------------------------

    longest_losing_streak = 0
    current_losing_streak = 0

    for value in r:

        if value < 0:

            current_losing_streak += 1

            longest_losing_streak = max(
                longest_losing_streak,
                current_losing_streak,
            )

        else:

            current_losing_streak = 0

    return {
        "trades": len(trades),

        "win_rate": (
            r > 0
        ).mean(),

        "avg_r": r.mean(),

        "median_r": r.median(),

        "total_r": r.sum(),

        "profit_factor": profit_factor,

        "max_drawdown_r": max_drawdown_r,

        "trade_sharpe": trade_sharpe,

        "long_trades": len(longs),

        "long_avg_r": long_avg_r,

        "short_trades": len(shorts),

        "short_avg_r": short_avg_r,

        "longest_losing_streak":
            longest_losing_streak,
    }


# ============================================================
# RUN ONE CONFIGURATION
# ============================================================

def run_configuration(
    sessions,
    or_minutes,
    target_r,
    signal_cutoff,
):

    trades = []

    for day in sessions:

        trade = process_day_parameterized(
            day=day,
            or_minutes=or_minutes,
            target_r=target_r,
            signal_cutoff=signal_cutoff,
        )

        if trade is not None:
            trades.append(trade)

    if not trades:
        return None

    trades_df = pd.DataFrame(trades)

    metrics = calculate_metrics(
        trades_df
    )

    return metrics


# ============================================================
# PARAMETER SWEEP
# ============================================================

def run_parameter_sweep():

    print(
        "\n========== PARAMETER SWEEP =========="
    )

    df = load_development_data()

    print(
        "Development period:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    # Prepare sessions once rather than
    # grouping the dataframe 120 times.
    sessions = [
        day.copy()
        for _, day
        in df.groupby(
            "date",
            sort=True,
        )
    ]

    total_configs = (
        len(OR_MINUTES)
        * len(TARGET_RS)
        * len(SIGNAL_CUTOFFS)
    )

    print(
        "Configurations:",
        total_configs,
    )

    results = []

    config_number = 0

    for or_minutes in OR_MINUTES:

        for target_r in TARGET_RS:

            for signal_cutoff in SIGNAL_CUTOFFS:

                config_number += 1

                print(
                    f"[{config_number:03d}/{total_configs}] "
                    f"OR={or_minutes:>2}m | "
                    f"Target={target_r:.1f}R | "
                    f"Cutoff={signal_cutoff.strftime('%H:%M')}"
                )

                metrics = run_configuration(
                    sessions=sessions,
                    or_minutes=or_minutes,
                    target_r=target_r,
                    signal_cutoff=signal_cutoff,
                )

                if metrics is None:
                    continue

                results.append({
                    "or_minutes": or_minutes,
                    "target_r": target_r,
                    "signal_cutoff":
                        signal_cutoff.strftime(
                            "%H:%M"
                        ),
                    **metrics,
                })

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        RESULTS_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    ranked = results_df.sort_values(
        "avg_r",
        ascending=False,
    )

    print()
    print(
        "========== TOP 15 BY AVG R =========="
    )

    columns = [
        "or_minutes",
        "target_r",
        "signal_cutoff",
        "trades",
        "win_rate",
        "avg_r",
        "total_r",
        "profit_factor",
        "max_drawdown_r",
        "trade_sharpe",
        "long_avg_r",
        "short_avg_r",
    ]

    print(
        ranked[
            columns
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print()
    print(
        "Results saved to:"
    )

    print(RESULTS_FILE)

    return results_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_parameter_sweep()