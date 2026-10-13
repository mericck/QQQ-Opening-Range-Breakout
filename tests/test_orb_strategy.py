from datetime import time

import numpy as np
import pandas as pd

from src.orb_strategy import (
    TARGET_R,
    DEVELOPMENT_START,
    DEVELOPMENT_END,
    get_opening_range,
    find_breakout,
    get_entry,
)


# ============================================================
# HELPERS
# ============================================================

def make_bar(
    timestamp,
    open_,
    high,
    low,
    close,
    volume=1000,
):
    return {
        "timestamp": pd.Timestamp(
            timestamp,
            tz="America/New_York",
        ),
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }


# ============================================================
# 1. BASELINE CONFIGURATION
# ============================================================

def test_target_r_is_frozen_at_2():
    """
    Baseline v1.0 uses a fixed 2R target.
    """
    assert TARGET_R == 2.0


def test_development_period_ends_before_oos():
    """
    Development data must never include 2023+ observations.
    """
    assert DEVELOPMENT_START.year == 2017
    assert DEVELOPMENT_END.year == 2022


# ============================================================
# 2. OPENING RANGE
# ============================================================

def test_opening_range_uses_first_three_bars():

    day = pd.DataFrame([
        make_bar(
            "2017-01-03 09:30",
            100.0,
            101.0,
            99.5,
            100.5,
        ),
        make_bar(
            "2017-01-03 09:35",
            100.5,
            102.0,
            100.0,
            101.5,
        ),
        make_bar(
            "2017-01-03 09:40",
            101.5,
            101.8,
            98.5,
            100.0,
        ),
        make_bar(
            "2017-01-03 09:45",
            100.0,
            105.0,
            97.0,
            104.0,
        ),
    ])

    result = get_opening_range(day)

    assert result["or_high"] == 102.0
    assert result["or_low"] == 98.5

# ============================================================
# 3. STRICT BREAKOUT LOGIC
# ============================================================

def test_close_equal_to_or_high_is_not_breakout():

    day = pd.DataFrame([
        make_bar(
            "2017-01-03 09:45",
            99.5,
            100.5,
            99.0,
            100.0,
        ),
        make_bar(
            "2017-01-03 09:50",
            100.0,
            101.0,
            99.8,
            100.2,
        ),
    ])

    result = find_breakout(
        day,
        or_high=100.0,
        or_low=90.0,
    )

    assert result is not None

    signal_idx, direction = result

    assert direction == "long"
    assert day.loc[signal_idx, "timestamp"].time() == time(9, 50)


def test_close_equal_to_or_low_is_not_breakout():

    day = pd.DataFrame([
        make_bar(
            "2017-01-03 09:45",
            95.0,
            96.0,
            89.5,
            90.0,
        ),
        make_bar(
            "2017-01-03 09:50",
            90.0,
            90.5,
            89.0,
            89.8,
        ),
    ])

    result = find_breakout(
        day,
        or_high=100.0,
        or_low=90.0,
    )

    assert result is not None

    signal_idx, direction = result

    assert direction == "short"
    assert day.loc[signal_idx, "timestamp"].time() == time(9, 50)


# ============================================================
# 4. FIRST VALID BREAKOUT ONLY
# ============================================================

def test_first_valid_breakout_is_selected():

    day = pd.DataFrame([
        make_bar(
            "2017-01-03 09:45",
            99.0,
            99.5,
            98.0,
            99.0,
        ),
        make_bar(
            "2017-01-03 09:50",
            100.0,
            101.5,
            99.5,
            101.0,
        ),
        make_bar(
            "2017-01-03 09:55",
            101.0,
            102.0,
            100.0,
            101.5,
        ),
    ])

    result = find_breakout(
        day,
        or_high=100.0,
        or_low=90.0,
    )

    signal_idx, direction = result

    assert direction == "long"
    assert day.loc[signal_idx, "timestamp"].time() == time(9, 50)


# ============================================================
# 5. NEXT-BAR OPEN ENTRY
# ============================================================

def test_entry_uses_next_bar_open():

    day = pd.DataFrame([
        make_bar(
            "2017-01-03 09:45",
            100.0,
            101.0,
            99.0,
            100.5,
        ),
        make_bar(
            "2017-01-03 09:50",
            100.5,
            102.0,
            100.0,
            101.5,
        ),
        make_bar(
            "2017-01-03 09:55",
            101.7,
            103.0,
            101.0,
            102.5,
        ),
    ])

    signal_idx = day.index[1]

    result = get_entry(
        day,
        signal_idx,
    )

    assert result is not None

    entry_idx, entry_bar = result

    assert entry_idx == day.index[2]
    assert entry_bar["timestamp"].time() == time(9, 55)
    assert entry_bar["open"] == 101.7


# ============================================================
# 6. REAL DEVELOPMENT RESULTS
# ============================================================

def test_development_trade_file_integrity():

    trades = pd.read_csv(
        "results/tables/orb_trades_development.csv"
    )

    assert len(trades) > 0

    trades["date"] = pd.to_datetime(
        trades["date"]
    )

    # OOS must remain untouched.
    assert trades["date"].dt.year.max() <= 2022

    # One trade maximum per session.
    assert trades["date"].duplicated().sum() == 0

    # Every trade must have an exit.
    assert trades["exit_price"].notna().all()
    assert trades["exit_reason"].notna().all()

    # Only expected exit types.
    allowed_exits = {
        "stop",
        "stop_ambiguous",
        "target",
        "time_exit",
        "early_close",
    }

    assert set(
        trades["exit_reason"].unique()
    ).issubset(allowed_exits)


# ============================================================
# 7. STOP / TARGET MATHEMATICS
# ============================================================

def test_stop_and_target_math():

    trades = pd.read_csv(
        "results/tables/orb_trades_development.csv"
    )

    longs = trades[
        trades["direction"] == "long"
    ].copy()

    shorts = trades[
        trades["direction"] == "short"
    ].copy()

    # LONG
    expected_long_risk = (
        longs["entry_price"]
        - longs["stop_price"]
    )

    assert np.allclose(
        longs["initial_risk"],
        expected_long_risk,
    )

    expected_long_target = (
        longs["entry_price"]
        + TARGET_R * expected_long_risk
    )

    assert np.allclose(
        longs["target_price"],
        expected_long_target,
    )

    # SHORT
    expected_short_risk = (
        shorts["stop_price"]
        - shorts["entry_price"]
    )

    assert np.allclose(
        shorts["initial_risk"],
        expected_short_risk,
    )

    expected_short_target = (
        shorts["entry_price"]
        - TARGET_R * expected_short_risk
    )

    assert np.allclose(
        shorts["target_price"],
        expected_short_target,
    )


# ============================================================
# 8. R-MULTIPLE MATHEMATICS
# ============================================================

def test_r_multiple_math():

    trades = pd.read_csv(
        "results/tables/orb_trades_development.csv"
    )

    expected_r = (
        trades["pnl_price"]
        / trades["initial_risk"]
    )

    assert np.allclose(
        trades["r_multiple"],
        expected_r,
    )


# ============================================================
# 9. BASELINE TRADE COUNT REGRESSION TEST
# ============================================================

def test_baseline_trade_count():

    trades = pd.read_csv(
        "results/tables/orb_trades_development.csv"
    )

    assert len(trades) == 1472


# ============================================================
# 10. EARLY-CLOSE REGRESSION TEST
# ============================================================

def test_early_close_trades_exist():

    trades = pd.read_csv(
        "results/tables/orb_trades_development.csv"
    )

    early_close = trades[
        trades["exit_reason"] == "early_close"
    ]

    assert len(early_close) == 8