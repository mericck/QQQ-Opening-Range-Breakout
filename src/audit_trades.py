from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

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

AUDIT_FILE = (
    PROJECT_ROOT
    / "results"
    / "tables"
    / "orb_full_sample_audit.csv"
)


# ============================================================
# STRATEGY CONSTANTS
# ============================================================

OR_START = time(9, 30)
OR_END = time(9, 45)

SIGNAL_START = time(9, 45)
SIGNAL_END = time(12, 0)

FORCED_EXIT_TIME = time(15, 0)

TARGET_R = 2.0

TOL = 1e-8


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    bars = pd.read_parquet(DATA_FILE)

    bars["timestamp"] = (
        pd.to_datetime(
            bars["timestamp"],
            utc=True,
        )
        .dt.tz_convert("America/New_York")
    )

    bars["date"] = bars["timestamp"].dt.date

    bars = bars.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    trades = pd.read_csv(TRADES_FILE)

    trades["date"] = pd.to_datetime(
        trades["date"]
    ).dt.date

    for col in [
        "signal_time",
        "entry_time",
        "exit_time",
    ]:
        trades[col] = (
            pd.to_datetime(
                trades[col],
                utc=True,
            )
            .dt.tz_convert("America/New_York")
        )

    return bars, trades


# ============================================================
# AUDIT ONE TRADE
# ============================================================

def audit_trade(trade, day):

    issues = []

    direction = trade["direction"]

    # --------------------------------------------------------
    # 1. OPENING RANGE
    # --------------------------------------------------------

    opening_range = day[
        (day["timestamp"].dt.time >= OR_START)
        & (day["timestamp"].dt.time < OR_END)
    ]

    if len(opening_range) != 3:

        issues.append(
            f"opening_range_bar_count={len(opening_range)}"
        )

        return {
            "audit_pass": False,
            "issues": "; ".join(issues),
        }

    expected_or_high = opening_range["high"].max()
    expected_or_low = opening_range["low"].min()

    if not np.isclose(
        trade["or_high"],
        expected_or_high,
        atol=TOL,
    ):
        issues.append("incorrect_or_high")

    if not np.isclose(
        trade["or_low"],
        expected_or_low,
        atol=TOL,
    ):
        issues.append("incorrect_or_low")

    # --------------------------------------------------------
    # 2. SIGNAL
    # --------------------------------------------------------

    signal_candidates = day[
        (day["timestamp"].dt.time >= SIGNAL_START)
        & (day["timestamp"].dt.time <= SIGNAL_END)
    ].copy()

    first_signal = None

    for _, bar in signal_candidates.iterrows():

        if bar["close"] > expected_or_high:

            first_signal = (
                bar["timestamp"],
                "long",
            )

            break

        if bar["close"] < expected_or_low:

            first_signal = (
                bar["timestamp"],
                "short",
            )

            break

    if first_signal is None:

        issues.append(
            "trade_exists_without_valid_signal"
        )

    else:

        expected_signal_time = first_signal[0]
        expected_direction = first_signal[1]

        if trade["signal_time"] != expected_signal_time:
            issues.append("incorrect_signal_time")

        if direction != expected_direction:
            issues.append("incorrect_direction")

    # --------------------------------------------------------
    # 3. ENTRY
    # --------------------------------------------------------

    signal_positions = day.index[
        day["timestamp"] == trade["signal_time"]
    ].tolist()

    if len(signal_positions) != 1:

        issues.append(
            "signal_bar_not_unique"
        )

    else:

        signal_idx = signal_positions[0]

        next_positions = day.index[
            day.index > signal_idx
        ]

        if len(next_positions) == 0:

            issues.append(
                "missing_next_entry_bar"
            )

        else:

            entry_idx = next_positions[0]
            expected_entry_bar = day.loc[entry_idx]

            if (
                trade["entry_time"]
                != expected_entry_bar["timestamp"]
            ):
                issues.append(
                    "incorrect_entry_time"
                )

            if not np.isclose(
                trade["entry_price"],
                expected_entry_bar["open"],
                atol=TOL,
            ):
                issues.append(
                    "incorrect_entry_price"
                )

    # --------------------------------------------------------
    # 4. STOP / TARGET MATH
    # --------------------------------------------------------

    entry = trade["entry_price"]

    if direction == "long":

        expected_stop = expected_or_low
        expected_risk = entry - expected_stop
        expected_target = (
            entry
            + TARGET_R * expected_risk
        )

    else:

        expected_stop = expected_or_high
        expected_risk = expected_stop - entry
        expected_target = (
            entry
            - TARGET_R * expected_risk
        )

    if expected_risk <= 0:
        issues.append(
            "non_positive_initial_risk"
        )

    if not np.isclose(
        trade["stop_price"],
        expected_stop,
        atol=TOL,
    ):
        issues.append("incorrect_stop")

    if not np.isclose(
        trade["initial_risk"],
        expected_risk,
        atol=TOL,
    ):
        issues.append("incorrect_initial_risk")

    if not np.isclose(
        trade["target_price"],
        expected_target,
        atol=TOL,
    ):
        issues.append("incorrect_target")

    # --------------------------------------------------------
    # 5. TRADE PATH
    # --------------------------------------------------------

    trade_path = day[
        (day["timestamp"] >= trade["entry_time"])
        & (day["timestamp"] <= trade["exit_time"])
    ].copy()

    if trade_path.empty:

        issues.append("empty_trade_path")

        return {
            "audit_pass": False,
            "issues": "; ".join(issues),
        }

    exit_rows = trade_path[
        trade_path["timestamp"]
        == trade["exit_time"]
    ]

    if len(exit_rows) != 1:

        issues.append(
            "exit_bar_not_unique"
        )

        return {
            "audit_pass": False,
            "issues": "; ".join(issues),
        }

    exit_bar = exit_rows.iloc[0]

    before_exit = trade_path[
        trade_path["timestamp"]
        < trade["exit_time"]
    ]

    stop = trade["stop_price"]
    target = trade["target_price"]

    # --------------------------------------------------------
    # 6. NO EARLIER STOP / TARGET
    # --------------------------------------------------------

    if direction == "long":

        earlier_stop = (
            before_exit["low"] <= stop
        ).any()

        earlier_target = (
            before_exit["high"] >= target
        ).any()

    else:

        earlier_stop = (
            before_exit["high"] >= stop
        ).any()

        earlier_target = (
            before_exit["low"] <= target
        ).any()

    if earlier_stop:
        issues.append(
            "stop_touched_before_recorded_exit"
        )

    if earlier_target:
        issues.append(
            "target_touched_before_recorded_exit"
        )

    # --------------------------------------------------------
    # 7. EXIT REASON
    # --------------------------------------------------------

    exit_reason = trade["exit_reason"]

    if exit_reason in {
        "stop",
        "stop_ambiguous",
    }:

        if direction == "long":
            stop_touched = (
                exit_bar["low"] <= stop
            )
        else:
            stop_touched = (
                exit_bar["high"] >= stop
            )

        if not stop_touched:
            issues.append(
                "stop_not_touched_on_exit_bar"
            )

    elif exit_reason == "target":

        if direction == "long":
            target_touched = (
                exit_bar["high"] >= target
            )
        else:
            target_touched = (
                exit_bar["low"] <= target
            )

        if not target_touched:
            issues.append(
                "target_not_touched_on_exit_bar"
            )

    elif exit_reason == "time_exit":

        if (
            exit_bar["timestamp"].time()
            != FORCED_EXIT_TIME
        ):
            issues.append(
                "incorrect_time_exit_time"
            )

        if not np.isclose(
            trade["exit_price"],
            exit_bar["open"],
            atol=TOL,
        ):
            issues.append(
                "incorrect_time_exit_price"
            )

    elif exit_reason == "early_close":

        final_bar = day.iloc[-1]

        if (
            trade["exit_time"]
            != final_bar["timestamp"]
        ):
            issues.append(
                "early_close_not_final_bar"
            )

        if not np.isclose(
            trade["exit_price"],
            final_bar["close"],
            atol=TOL,
        ):
            issues.append(
                "incorrect_early_close_price"
            )

    else:

        issues.append(
            f"unknown_exit_reason={exit_reason}"
        )

    # --------------------------------------------------------
    # 8. INTRABAR AMBIGUITY
    # --------------------------------------------------------

    if direction == "long":

        both_touched = (
            exit_bar["low"] <= stop
            and exit_bar["high"] >= target
        )

    else:

        both_touched = (
            exit_bar["high"] >= stop
            and exit_bar["low"] <= target
        )

    if both_touched:

        if exit_reason != "stop_ambiguous":
            issues.append(
                "unflagged_intrabar_ambiguity"
            )

    # --------------------------------------------------------
    # 9. GAP-THROUGH STOP
    # --------------------------------------------------------

    gap_through_stop = False
    stop_gap_amount = 0.0

    if exit_reason in {
        "stop",
        "stop_ambiguous",
    }:

        if direction == "long":

            if exit_bar["open"] < stop:

                gap_through_stop = True

                stop_gap_amount = (
                    stop
                    - exit_bar["open"]
                )

        else:

            if exit_bar["open"] > stop:

                gap_through_stop = True

                stop_gap_amount = (
                    exit_bar["open"]
                    - stop
                )

    # --------------------------------------------------------
    # 10. GAP-THROUGH TARGET
    # --------------------------------------------------------

    gap_through_target = False
    target_gap_amount = 0.0

    if exit_reason == "target":

        if direction == "long":

            if exit_bar["open"] > target:

                gap_through_target = True

                target_gap_amount = (
                    exit_bar["open"]
                    - target
                )

        else:

            if exit_bar["open"] < target:

                gap_through_target = True

                target_gap_amount = (
                    target
                    - exit_bar["open"]
                )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "audit_pass": len(issues) == 0,
        "issues": "; ".join(issues),
        "gap_through_stop": gap_through_stop,
        "stop_gap_amount": stop_gap_amount,
        "gap_through_target": gap_through_target,
        "target_gap_amount": target_gap_amount,
    }


# ============================================================
# FULL SAMPLE
# ============================================================

def run_full_audit():

    print("FULL-SAMPLE AUDIT STARTED")

    bars, trades = load_data()

    development_bars = bars[
        bars["timestamp"].dt.year <= 2022
    ].copy()

    bars_by_date = {
        date: group.copy()
        for date, group
        in development_bars.groupby("date")
    }

    results = []

    for _, trade in trades.iterrows():

        trade_date = trade["date"]

        if trade_date not in bars_by_date:

            results.append({
                "date": trade_date,
                "direction": trade["direction"],
                "audit_pass": False,
                "issues": "missing_session_data",
                "gap_through_stop": False,
                "stop_gap_amount": 0.0,
                "gap_through_target": False,
                "target_gap_amount": 0.0,
            })

            continue

        result = audit_trade(
            trade,
            bars_by_date[trade_date],
        )

        result["date"] = trade_date
        result["direction"] = trade["direction"]
        result["exit_reason"] = trade["exit_reason"]

        results.append(result)

    audit = pd.DataFrame(results)

    audit.to_csv(
        AUDIT_FILE,
        index=False,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("FULL-SAMPLE AUDIT REPORT")
    print("=" * 60)

    print(f"Trades audited: {len(audit):,}")

    passed = audit["audit_pass"].sum()

    failed = (
        ~audit["audit_pass"]
    ).sum()

    print(f"Passed: {passed:,}")
    print(f"Failed: {failed:,}")

    print()

    if failed > 0:

        print("FAILED AUDITS")
        print("-" * 60)

        failed_rows = audit[
            ~audit["audit_pass"]
        ]

        print(
            failed_rows[
                [
                    "date",
                    "direction",
                    "exit_reason",
                    "issues",
                ]
            ].to_string(index=False)
        )

    else:

        print(
            "All trades passed structural audit."
        )

    # ========================================================
    # GAP REPORT
    # ========================================================

    print()
    print("=" * 60)
    print("EXECUTION EDGE CASES")
    print("=" * 60)

    gap_stops = audit[
        audit["gap_through_stop"] == True
    ]

    gap_targets = audit[
        audit["gap_through_target"] == True
    ]

    print(
        f"Gap-through stops: {len(gap_stops):,}"
    )

    print(
        f"Gap-through targets: {len(gap_targets):,}"
    )

    if len(gap_stops) > 0:

        print()
        print("STOP GAP SUMMARY")

        print(
            gap_stops["stop_gap_amount"]
            .describe()
        )

        print()
        print("Largest stop gaps:")

        print(
            gap_stops.sort_values(
                "stop_gap_amount",
                ascending=False,
            )
            .head(10)[
                [
                    "date",
                    "direction",
                    "stop_gap_amount",
                ]
            ]
            .to_string(index=False)
        )

    if len(gap_targets) > 0:

        print()
        print("TARGET GAP SUMMARY")

        print(
            gap_targets[
                "target_gap_amount"
            ].describe()
        )

    print()
    print(
        f"Audit saved to:\n{AUDIT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_full_audit()