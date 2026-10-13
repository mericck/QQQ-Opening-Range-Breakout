from pathlib import Path
from datetime import time

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "QQQ_5min_2017_2026_processed.parquet"
)

TRADES_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "orb_trades_development.csv"
)


# ============================================================
# SAMPLE DEFINITION
# ============================================================
#
# IMPORTANT:
# 2023-01-01 onward is reserved as the out-of-sample holdout.
# The strategy development process must not use OOS performance.
#
# Development sample:
# 2017-01-01 -> 2022-12-31
#
# OOS holdout:
# 2023-01-01 -> 2026-09-30
# ============================================================

DEVELOPMENT_START = pd.Timestamp(
    "2017-01-01 00:00:00",
    tz="America/New_York",
)

DEVELOPMENT_END = pd.Timestamp(
    "2022-12-31 23:59:59",
    tz="America/New_York",
)


# ============================================================
# STRATEGY PARAMETERS
# ============================================================

# Opening range:
# 09:30, 09:35 and 09:40 bars
OR_START = time(9, 30)
OR_END = time(9, 45)

# Breakout signals are searched from 09:45 through 12:00.
SIGNAL_START = time(9, 45)
SIGNAL_END = time(12, 0)

# Any position still open at 15:00 is closed.
FORCED_EXIT_TIME = time(15, 0)

# Fixed reward-to-risk target
TARGET_R = 2


# ============================================================
# DATA LOADING
# ============================================================

def load_development_data():
    """
    Load cleaned QQQ 5-minute data and return ONLY
    the development sample.

    The OOS period is intentionally excluded here.
    """

    df = pd.read_parquet(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        utc=True,
    ).dt.tz_convert("America/New_York")

    df = (
        df
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Hard development-sample filter
    # --------------------------------------------------------

    df = df[
        (df["timestamp"] >= DEVELOPMENT_START)
        & (df["timestamp"] <= DEVELOPMENT_END)
    ].copy()

    if df.empty:
        raise ValueError(
            "Development dataset is empty."
        )

    # --------------------------------------------------------
    # Safety check:
    # OOS observations must never enter this backtest.
    # --------------------------------------------------------

    if (df["timestamp"].dt.year >= 2023).any():
        raise ValueError(
            "OOS DATA DETECTED. "
            "Development backtest aborted."
        )

    return df


# ============================================================
# OPENING RANGE
# ============================================================

def get_opening_range(day):
    """
    Calculate the opening range from the first
    three 5-minute bars:

    09:30
    09:35
    09:40

    ORH = highest high
    ORL = lowest low
    """

    opening_range = day[
        (day["timestamp"].dt.time >= OR_START)
        & (day["timestamp"].dt.time < OR_END)
    ]

    if len(opening_range) != 3:
        return None

    return {
        "or_high": opening_range["high"].max(),
        "or_low": opening_range["low"].min(),
    }


# ============================================================
# BREAKOUT DETECTION
# ============================================================

def find_breakout(day, or_high, or_low):
    """
    Find the FIRST valid breakout of the day.

    Long:
        candle close > ORH

    Short:
        candle close < ORL

    Signals are considered only between
    09:45 and 12:00.

    Maximum one trade per day.
    """

    signal_bars = day[
        (day["timestamp"].dt.time >= SIGNAL_START)
        & (day["timestamp"].dt.time <= SIGNAL_END)
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

def get_entry(day, signal_idx):
    """
    Enter at the OPEN of the 5-minute candle
    immediately following the breakout signal.

    This avoids entering at the same close that
    was required to confirm the breakout.
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
# TRADE SIMULATION
# ============================================================

def simulate_trade(
    day,
    entry_idx,
    direction,
    stop_price,
    target_price,
):
    """
    Simulate a trade bar-by-bar from the entry bar.

    Normal trading day:
        Position is force-closed at 15:00 ET.

    Early-close trading day:
        If the market closes before 15:00 ET,
        the position is closed at the close of the
        final available RTH candle.

    Intrabar ambiguity:
        If both stop and target are touched within
        the same 5-minute candle, stop is assumed
        to occur first.
    """

    trade_bars = day.loc[entry_idx:]

    # Last available RTH bar of the session.
    # Needed for official early-close trading days.
    last_idx = trade_bars.index[-1]

    for idx, bar in trade_bars.iterrows():

        bar_time = bar["timestamp"].time()

        # ====================================================
        # NORMAL 15:00 FORCED EXIT
        # ====================================================
        #
        # At exactly 15:00, we exit at the OPEN of the
        # 15:00 candle. Therefore we must NOT use the
        # high/low of that candle.
        # ====================================================

        if bar_time >= FORCED_EXIT_TIME:

            return {
                "exit_idx": idx,
                "exit_price": bar["open"],
                "exit_reason": "time_exit",
            }

        # ====================================================
        # LONG POSITION
        # ====================================================

        if direction == "long":

            stop_hit = (
                bar["low"] <= stop_price
            )

            target_hit = (
                bar["high"] >= target_price
            )

            # Both stop and target touched
            # within the same 5-minute candle.
            if stop_hit and target_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": stop_price,
                    "exit_reason": "stop_ambiguous",
                }

            if stop_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": stop_price,
                    "exit_reason": "stop",
                }

            if target_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": target_price,
                    "exit_reason": "target",
                }

        # ====================================================
        # SHORT POSITION
        # ====================================================

        elif direction == "short":

            stop_hit = (
                bar["high"] >= stop_price
            )

            target_hit = (
                bar["low"] <= target_price
            )

            # Both stop and target touched
            # within the same 5-minute candle.
            if stop_hit and target_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": stop_price,
                    "exit_reason": "stop_ambiguous",
                }

            if stop_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": stop_price,
                    "exit_reason": "stop",
                }

            if target_hit:

                return {
                    "exit_idx": idx,
                    "exit_price": target_price,
                    "exit_reason": "target",
                }

        else:

            raise ValueError(
                f"Unknown direction: {direction}"
            )

        # ====================================================
        # EARLY-CLOSE SESSION EXIT
        # ====================================================
        #
        # This check MUST come after the stop/target checks.
        #
        # Example:
        # On a 13:00 early-close day, the final candle starts
        # at 12:55. We first check whether stop or target was
        # touched during 12:55-13:00.
        #
        # If neither was touched, we exit at that candle's
        # CLOSE, representing the official market close.
        # ====================================================

        if idx == last_idx:

            return {
                "exit_idx": idx,
                "exit_price": bar["close"],
                "exit_reason": "early_close",
            }

    return None


# ============================================================
# PROCESS ONE SESSION
# ============================================================

def process_day(day):
    """
    Process one complete trading session.

    Maximum:
        one trade per day.
    """

    day = day.sort_values(
        "timestamp"
    )

    # --------------------------------------------------------
    # 1. Opening range
    # --------------------------------------------------------

    opening_range = get_opening_range(day)

    if opening_range is None:
        return None

    or_high = opening_range["or_high"]
    or_low = opening_range["or_low"]

    # --------------------------------------------------------
    # 2. First valid breakout
    # --------------------------------------------------------

    breakout = find_breakout(
        day=day,
        or_high=or_high,
        or_low=or_low,
    )

    if breakout is None:
        return None

    signal_idx, direction = breakout

    signal_bar = day.loc[signal_idx]

    # --------------------------------------------------------
    # 3. Entry on following candle open
    # --------------------------------------------------------

    entry = get_entry(
        day=day,
        signal_idx=signal_idx,
    )

    if entry is None:
        return None

    entry_idx, entry_bar = entry

    entry_price = entry_bar["open"]

    # --------------------------------------------------------
    # 4. Stop, risk and target
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
            + TARGET_R * initial_risk
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
            - TARGET_R * initial_risk
        )

    else:

        raise ValueError(
            f"Unknown direction: {direction}"
        )

    # --------------------------------------------------------
    # 5. Simulate trade
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
    # 6. Calculate PnL in price units
    # --------------------------------------------------------

    if direction == "long":

        pnl = (
            exit_price - entry_price
        )

    else:

        pnl = (
            entry_price - exit_price
        )

    # --------------------------------------------------------
    # 7. R-multiple
    # --------------------------------------------------------

    r_multiple = (
        pnl / initial_risk
    )

    # --------------------------------------------------------
    # 8. Trade record
    # --------------------------------------------------------

    return {
        "date": day["date"].iloc[0],

        "direction": direction,

        "or_high": or_high,
        "or_low": or_low,

        "signal_time": signal_bar["timestamp"],
        "signal_close": signal_bar["close"],

        "entry_time": entry_bar["timestamp"],
        "entry_price": entry_price,

        "stop_price": stop_price,
        "target_price": target_price,
        "initial_risk": initial_risk,

        "exit_time": exit_bar["timestamp"],
        "exit_price": exit_price,
        "exit_reason": trade_result["exit_reason"],

        "pnl_price": pnl,
        "r_multiple": r_multiple,
    }


# ============================================================
# DEVELOPMENT BACKTEST
# ============================================================

def run_development_backtest():
    """
    Run the strategy ONLY on the development sample.

    OOS data is deliberately excluded.
    """

    df = load_development_data()

    print(
        "\n========== DEVELOPMENT SAMPLE =========="
    )

    print(
        "Backtest period:",
        df["timestamp"].min(),
        "->",
        df["timestamp"].max(),
    )

    print(
        "Trading sessions:",
        f"{df['date'].nunique():,}",
    )

    print(
        "Rows:",
        f"{len(df):,}",
    )

    # --------------------------------------------------------
    # Generate trades
    # --------------------------------------------------------

    trades = []

    for session_date, day in df.groupby(
        "date",
        sort=True,
    ):

        trade = process_day(day)

        if trade is not None:
            trades.append(trade)

    trades_df = pd.DataFrame(trades)

    if trades_df.empty:

        print(
            "\nNo trades were generated."
        )

        return trades_df

    # --------------------------------------------------------
    # Final OOS safety check
    # --------------------------------------------------------

    trade_dates = pd.to_datetime(
        trades_df["date"]
    )

    if (
        trade_dates
        >= pd.Timestamp("2023-01-01")
    ).any():

        raise ValueError(
            "OOS TRADE DETECTED. "
            "Backtest aborted."
        )

    # --------------------------------------------------------
    # Save development trades
    # --------------------------------------------------------

    TRADES_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    trades_df.to_csv(
        TRADES_FILE,
        index=False,
    )

    # --------------------------------------------------------
    # Preliminary diagnostics
    #
    # These are NOT final research conclusions.
    # Their purpose is to help validate the engine.
    # --------------------------------------------------------

    print(
        "\n========== ORB DEVELOPMENT BACKTEST =========="
    )

    print(
        "Trades generated:",
        f"{len(trades_df):,}",
    )

    print("\nDirection:")

    print(
        trades_df[
            "direction"
        ].value_counts()
    )

    print("\nExit reasons:")

    print(
        trades_df[
            "exit_reason"
        ].value_counts()
    )

    ambiguous_count = (
        trades_df["exit_reason"]
        == "stop_ambiguous"
    ).sum()

    print(
        "\nAmbiguous stop/target bars:",
        f"{ambiguous_count:,}",
    )

    print(
        "Ambiguous rate:",
        f"{ambiguous_count / len(trades_df):.2%}",
    )

    print(
        "\nAverage R:",
        round(
            trades_df[
                "r_multiple"
            ].mean(),
            4,
        ),
    )

    print(
        "Total R:",
        round(
            trades_df[
                "r_multiple"
            ].sum(),
            2,
        ),
    )

    print(
        "Win rate:",
        f"{(
            trades_df['r_multiple'] > 0
        ).mean():.2%}",
    )

    print(
        "\nTrades saved to:"
    )

    print(TRADES_FILE)

    return trades_df


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    run_development_backtest()