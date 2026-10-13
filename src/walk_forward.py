
"""Rolling walk-forward validation for the QQQ ORB strategy.

Training: 2 years
Forward validation: 1 year
Development sample: 2017-2022

The 2023-2026 final OOS holdout is not accessed.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.orb_strategy import load_development_data

from src.parameter_sweep import (
    OR_MINUTES,
    TARGET_RS,
    SIGNAL_CUTOFFS,
    process_day_parameterized,
    calculate_metrics,
)


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_ROOT / "results" / "tables"

FOLDS = [
    (2017, 2018, 2019),
    (2018, 2019, 2020),
    (2019, 2020, 2021),
    (2020, 2021, 2022),
]

BASELINE_OR = 15
BASELINE_TARGET = 2.0
BASELINE_CUTOFF = "12:00"


# ============================================================
# GENERATE TRADES FOR A GIVEN CONFIGURATION
# ============================================================

def generate_trades(
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

    return pd.DataFrame(trades)


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def evaluate(trades):

    if trades.empty:
        return None

    return calculate_metrics(trades)


# ============================================================
# MAIN WALK-FORWARD FUNCTION
# ============================================================

def run_walk_forward():

    print("\n========== WALK-FORWARD VALIDATION ==========")

    # Load only development data.
    df = load_development_data()

    if (df["timestamp"].dt.year >= 2023).any():
        raise RuntimeError("OOS data detected!")

    # Verify that the full parameter grid is available.
    total_configs = (
        len(OR_MINUTES)
        * len(TARGET_RS)
        * len(SIGNAL_CUTOFFS)
    )

    if total_configs != 120:
        raise ValueError(
            f"Expected 120 configurations, got {total_configs}."
        )

    # Organize sessions by year.
    sessions_by_year = {}

    for _, day in df.groupby("date", sort=True):

        year = int(day["timestamp"].iloc[0].year)

        sessions_by_year.setdefault(
            year, []
        ).append(day.copy())

    fold_results = []
    forward_trade_records = []

    # ========================================================
    # WALK-FORWARD LOOP
    # ========================================================

    for fold_number, (
        train_start,
        train_end,
        test_year,
    ) in enumerate(FOLDS, start=1):

        print(
            f"\nFOLD {fold_number}/{len(FOLDS)}"
            f" | Train: {train_start}-{train_end}"
            f" | Test: {test_year}",
            flush=True,
        )

        # ----------------------------------------------------
        # Split training and test sessions
        # ----------------------------------------------------

        train_sessions = []

        for year in range(train_start, train_end + 1):
            train_sessions.extend(
                sessions_by_year.get(year, [])
            )

        test_sessions = sessions_by_year.get(
            test_year, []
        )

        if not train_sessions or not test_sessions:
            raise ValueError(
                f"Missing sessions in fold {fold_number}"
            )

        # ----------------------------------------------------
        # Optimize using TRAINING DATA ONLY
        # ----------------------------------------------------

        candidates = []

        for or_minutes in OR_MINUTES:

            for target_r in TARGET_RS:

                for cutoff in SIGNAL_CUTOFFS:

                    trades = generate_trades(
                        sessions=train_sessions,
                        or_minutes=or_minutes,
                        target_r=target_r,
                        signal_cutoff=cutoff,
                    )

                    metrics = evaluate(trades)

                    if metrics is None:
                        continue

                    if not np.isfinite(metrics["avg_r"]):
                        continue

                    candidates.append({
                        "or_minutes": or_minutes,
                        "target_r": target_r,
                        "signal_cutoff": cutoff,
                        "cutoff_str": cutoff.strftime("%H:%M"),
                        **metrics,
                    })

        if not candidates:
            raise RuntimeError(
                f"No valid configurations in fold {fold_number}"
            )

        # ----------------------------------------------------
        # Select the highest training Average R
        # ----------------------------------------------------

        candidates.sort(
            key=lambda x: (
                -x["avg_r"],
                -x["trades"],
                x["or_minutes"],
                x["target_r"],
                x["cutoff_str"],
            )
        )

        best = candidates[0]

        selected_or = best["or_minutes"]
        selected_target = best["target_r"]
        selected_cutoff = best["signal_cutoff"]

        # ----------------------------------------------------
        # Forward test: no parameter changes allowed
        # ----------------------------------------------------

        forward_trades = generate_trades(
            sessions=test_sessions,
            or_minutes=selected_or,
            target_r=selected_target,
            signal_cutoff=selected_cutoff,
        )

        forward_metrics = evaluate(forward_trades)

        # ----------------------------------------------------
        # Fixed baseline for comparison
        # ----------------------------------------------------

        baseline_cutoff = next(
            cutoff
            for cutoff in SIGNAL_CUTOFFS
            if cutoff.strftime("%H:%M") == BASELINE_CUTOFF
        )

        baseline_trades = generate_trades(
            sessions=test_sessions,
            or_minutes=BASELINE_OR,
            target_r=BASELINE_TARGET,
            signal_cutoff=baseline_cutoff,
        )

        baseline_metrics = evaluate(baseline_trades)

        if forward_metrics is None or baseline_metrics is None:
            raise RuntimeError(
                f"No forward trades in {test_year}"
            )

        # ----------------------------------------------------
        # Store fold-level results
        # ----------------------------------------------------

        for strategy, trades, metrics in [
            ("selected", forward_trades, forward_metrics),
            ("baseline", baseline_trades, baseline_metrics),
        ]:

            if strategy == "selected":
                current_or = selected_or
                current_target = selected_target
                current_cutoff = selected_cutoff.strftime("%H:%M")
            else:
                current_or = BASELINE_OR
                current_target = BASELINE_TARGET
                current_cutoff = BASELINE_CUTOFF

            fold_results.append({
                "fold": fold_number,
                "train_start": train_start,
                "train_end": train_end,
                "test_year": test_year,
                "strategy": strategy,
                "or_minutes": current_or,
                "target_r": current_target,
                "signal_cutoff": current_cutoff,
                "train_avg_r_selected": (
                    best["avg_r"]
                    if strategy == "selected"
                    else np.nan
                ),
                **metrics,
            })

            trade_records = trades.copy()

            trade_records["test_year"] = test_year
            trade_records["strategy"] = strategy
            trade_records["or_minutes"] = current_or
            trade_records["target_r"] = current_target
            trade_records["signal_cutoff"] = current_cutoff

            forward_trade_records.append(trade_records)

        print(
            f"Selected: {selected_or}m / "
            f"{selected_target}R / "
            f"{selected_cutoff.strftime('%H:%M')}"
        )

        print(
            f"Train Avg R: {best['avg_r']:.4f}"
        )

        print(
            f"Forward Avg R: {forward_metrics['avg_r']:.4f}"
        )

        print(
            f"Baseline Avg R: {baseline_metrics['avg_r']:.4f}"
        )

    # ========================================================
    # COMBINE RESULTS
    # ========================================================

    folds_df = pd.DataFrame(fold_results)

    trades_df = pd.concat(
        forward_trade_records,
        ignore_index=True,
    )

    if not trades_df["test_year"].between(2019, 2022).all():
        raise RuntimeError(
            "Forward test dates outside development sample!"
        )

    # Calculate aggregate metrics in chronological order.
    summary_rows = []

    for strategy in ["selected", "baseline"]:

        strategy_trades = (
            trades_df[trades_df["strategy"] == strategy]
            .sort_values("entry_time")
        )

        metrics = evaluate(strategy_trades)

        summary_rows.append({
            "strategy": strategy,
            **metrics,
        })

    summary_df = pd.DataFrame(summary_rows)

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    folds_df.to_csv(
        OUTPUT_DIR / "walk_forward_folds.csv",
        index=False,
    )

    trades_df.to_csv(
        OUTPUT_DIR / "walk_forward_trades.csv",
        index=False,
    )

    summary_df.to_csv(
        OUTPUT_DIR / "walk_forward_summary.csv",
        index=False,
    )

    # ========================================================
    # PRINT SUMMARY
    # ========================================================

    print("\n========== FORWARD RESULTS ==========")

    print(
        summary_df[
            [
                "strategy",
                "trades",
                "win_rate",
                "avg_r",
                "total_r",
                "profit_factor",
                "max_drawdown_r",
            ]
        ].to_string(index=False)
    )

    print("\nResults saved to:", OUTPUT_DIR)

    return folds_df, trades_df, summary_df


if __name__ == "__main__":
    run_walk_forward()
