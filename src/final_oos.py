
"""Final out-of-sample evaluation: QQQ ORB, 2023-2026."""

from datetime import time
from pathlib import Path
import itertools

import numpy as np
import pandas as pd

from src.parameter_sweep import (
    OR_MINUTES,
    TARGET_RS,
    SIGNAL_CUTOFFS,
    process_day_parameterized,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data/processed/QQQ_5min_2017_2026_processed.parquet"
OUTPUT = ROOT / "results/tables"

OOS_START = pd.Timestamp("2023-01-01", tz="America/New_York")
OOS_END = pd.Timestamp("2026-10-01", tz="America/New_York")

PRIMARY = (10, 3.0, time(10, 30))
BASELINE = (15, 2.0, time(12, 0))

GRID = list(itertools.product(
    OR_MINUTES, TARGET_RS, SIGNAL_CUTOFFS
))

assert len(GRID) == 120
assert PRIMARY in GRID
assert BASELINE in GRID


def load_data():
    """Load processed data without the development-only filter."""

    df = pd.read_parquet(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"], utc=True
    ).dt.tz_convert("America/New_York")

    df = df.sort_values("timestamp").copy()

    # All training and testing stays within the downloaded range.
    df = df[df["timestamp"] < OOS_END].copy()

    if df.empty:
        raise ValueError("Empty dataset")

    if df["timestamp"].max() < pd.Timestamp(
        "2026-09-30 15:55",
        tz="America/New_York"
    ):
        raise ValueError("Final OOS data appears incomplete")

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps detected")

    return df


def organize_sessions(df):
    """Group individual trading sessions by calendar year."""

    sessions = {}

    for _, day in df.groupby("date", sort=True):
        year = int(day["timestamp"].iloc[0].year)
        sessions.setdefault(year, []).append(day.copy())

    return sessions


def generate_trades(days, params):
    """Apply an unchanged strategy configuration."""

    or_minutes, target_r, cutoff = params
    records = []

    for day in days:
        trade = process_day_parameterized(
            day=day,
            or_minutes=or_minutes,
            target_r=target_r,
            signal_cutoff=cutoff,
        )

        if trade is not None:
            records.append(trade)

    return pd.DataFrame(records)


def metrics(trades):
    """Evaluate chronological R-multiple performance."""

    if trades.empty:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_r": np.nan,
            "total_r": 0.0,
            "profit_factor": np.nan,
            "max_drawdown_r": np.nan,
        }

    trades = trades.sort_values("entry_time")
    r = trades["r_multiple"].astype(float)

    equity = np.r_[0.0, r.cumsum().to_numpy()]
    peaks = np.maximum.accumulate(equity)
    drawdown = equity - peaks

    profits = r[r > 0].sum()
    losses = abs(r[r < 0].sum())

    return {
        "trades": len(r),
        "win_rate": (r > 0).mean(),
        "avg_r": r.mean(),
        "total_r": r.sum(),
        "profit_factor": (
            profits / losses if losses > 0 else np.nan
        ),
        "max_drawdown_r": abs(drawdown.min()),
    }


def select_parameters(training_days):
    """Choose highest Average R using past data only."""

    candidates = []

    for params in GRID:
        trades = generate_trades(training_days, params)

        if trades.empty:
            continue

        score = trades["r_multiple"].mean()

        if not np.isfinite(score):
            continue

        candidates.append({
            "params": params,
            "avg_r": score,
            "trades": len(trades),
        })

    if not candidates:
        raise RuntimeError("No valid training configurations")

    candidates.sort(
        key=lambda x: (
            -x["avg_r"],
            -x["trades"],
            x["params"][0],
            x["params"][1],
            x["params"][2],
        )
    )

    return candidates[0]


def run():

    df = load_data()
    sessions = organize_sessions(df)

    all_records = []
    annual_rows = []
    selection_rows = []

    for year in range(2023, 2027):

        test_days = sessions.get(year, [])

        if not test_days:
            raise RuntimeError(f"No test data for {year}")

        # The dynamic benchmark uses only the previous
        # two completed calendar years.
        training_days = (
            sessions.get(year - 2, [])
            + sessions.get(year - 1, [])
        )

        if not training_days:
            raise RuntimeError(f"No training data for {year}")

        best = select_parameters(training_days)
        dynamic_params = best["params"]

        selection_rows.append({
            "test_year": year,
            "train_start": year - 2,
            "train_end": year - 1,
            "or_minutes": dynamic_params[0],
            "target_r": dynamic_params[1],
            "signal_cutoff": dynamic_params[2].strftime("%H:%M"),
            "train_avg_r": best["avg_r"],
            "train_trades": best["trades"],
        })

        configs = {
            "primary": PRIMARY,
            "baseline": BASELINE,
            "walk_forward": dynamic_params,
        }

        for strategy, params in configs.items():

            trades = generate_trades(test_days, params)

            if trades.empty:
                raise RuntimeError(
                    f"No trades: {strategy}, {year}"
                )

            trades["strategy"] = strategy
            trades["test_year"] = year

            all_records.append(trades)

            annual_rows.append({
                "strategy": strategy,
                "year": year,
                **metrics(trades),
            })

        print(
            f"Completed {year} | "
            f"WF selected: {dynamic_params}",
            flush=True,
        )

    combined = pd.concat(all_records, ignore_index=True)

    # Validate the final evaluation period.
    entry_times = pd.to_datetime(
        combined["entry_time"], utc=True
    ).dt.tz_convert("America/New_York")

    if not entry_times.between(
        OOS_START, OOS_END, inclusive="left"
    ).all():
        raise RuntimeError("Trades outside final OOS window")

    if combined.duplicated(
        ["strategy", "date"]
    ).any():
        raise RuntimeError("More than one trade per strategy/day")

    summary = []

    for strategy, group in combined.groupby("strategy"):
        summary.append({
            "strategy": strategy,
            **metrics(group),
        })

    summary_df = pd.DataFrame(summary)
    annual_df = pd.DataFrame(annual_rows)
    selections_df = pd.DataFrame(selection_rows)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    summary_df.to_csv(
        OUTPUT / "final_oos_summary.csv", index=False
    )

    annual_df.to_csv(
        OUTPUT / "final_oos_annual.csv", index=False
    )

    selections_df.to_csv(
        OUTPUT / "final_oos_selections.csv", index=False
    )

    combined.to_csv(
        OUTPUT / "final_oos_trades.csv", index=False
    )

    print("\n========== FINAL OOS RESULTS ==========")
    print(summary_df.to_string(index=False))

    print("\n========== ANNUAL TOTAL R ==========")
    print(
        annual_df.pivot(
            index="year",
            columns="strategy",
            values="total_r",
        ).round(2).to_string()
    )

    print("\n========== WALK-FORWARD SELECTIONS ==========")
    print(selections_df.to_string(index=False))

    return summary_df, annual_df, selections_df, combined


if __name__ == "__main__":
    run()
