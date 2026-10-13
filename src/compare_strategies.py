
"""
Compare three ORB approaches over 2019-2022:

1. Fixed baseline: 15m / 2R / 12:00
2. Fixed candidate: 10m / 3R / 10:30
3. Rolling walk-forward selected parameters

All comparisons are development-stage analyses.
"""

from pathlib import Path
from datetime import time

import numpy as np
import pandas as pd

from src.orb_strategy import load_development_data
from src.parameter_sweep import process_day_parameterized


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

WF_TRADES_FILE = OUTPUT / "walk_forward_trades.csv"

START_YEAR = 2019
END_YEAR = 2022

FIXED_CONFIGS = {
    "baseline": {
        "or_minutes": 15,
        "target_r": 2.0,
        "signal_cutoff": time(12, 0),
    },
    "fixed_candidate": {
        "or_minutes": 10,
        "target_r": 3.0,
        "signal_cutoff": time(10, 30),
    },
}


def generate_fixed_trades(sessions, name, config):
    """Run a fixed configuration on the same sessions."""

    records = []

    for day in sessions:
        trade = process_day_parameterized(
            day=day,
            **config,
        )

        if trade is not None:
            records.append(trade)

    trades = pd.DataFrame(records)

    if trades.empty:
        raise RuntimeError(f"No trades for {name}")

    trades["strategy"] = name

    return trades


def calculate_comparison_metrics(trades):
    """Calculate performance and risk metrics in R."""

    trades = trades.sort_values("entry_time").copy()

    r = trades["r_multiple"].astype(float)

    equity = np.r_[0.0, r.cumsum().to_numpy()]
    peaks = np.maximum.accumulate(equity)
    drawdowns = equity - peaks

    winners = r[r > 0]
    losers = r[r < 0]

    gross_profit = winners.sum()
    gross_loss = abs(losers.sum())

    profit_factor = (
        gross_profit / gross_loss
        if gross_loss > 0
        else np.nan
    )

    losing_streak = 0
    longest_streak = 0

    for value in r:
        if value < 0:
            losing_streak += 1
            longest_streak = max(
                longest_streak,
                losing_streak,
            )
        else:
            losing_streak = 0

    return {
        "trades": len(trades),
        "win_rate": (r > 0).mean(),
        "avg_r": r.mean(),
        "total_r": r.sum(),
        "profit_factor": profit_factor,
        "max_drawdown_r": abs(drawdowns.min()),
        "longest_losing_streak": longest_streak,
        "long_avg_r": trades.loc[
            trades["direction"] == "long",
            "r_multiple",
        ].mean(),
        "short_avg_r": trades.loc[
            trades["direction"] == "short",
            "r_multiple",
        ].mean(),
    }


def run_comparison():

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    # Development data only.
    df = load_development_data()

    df = df[
        df["timestamp"].dt.year.between(
            START_YEAR,
            END_YEAR,
        )
    ].copy()

    if df.empty or df["timestamp"].dt.year.max() > 2022:
        raise RuntimeError("Invalid development period")

    sessions = [
        day.copy()
        for _, day in df.groupby("date", sort=True)
    ]

    # Generate fixed strategies.
    all_trades = []

    for name, config in FIXED_CONFIGS.items():

        trades = generate_fixed_trades(
            sessions,
            name,
            config,
        )

        all_trades.append(trades)

    # Load existing walk-forward forward-test trades.
    wf = pd.read_csv(WF_TRADES_FILE)

    wf = wf[wf["strategy"] == "selected"].copy()

    wf["strategy"] = "walk_forward"

    if not wf["test_year"].between(
        START_YEAR, END_YEAR
    ).all():
        raise RuntimeError("Unexpected walk-forward dates")

    all_trades.append(wf)

    combined = pd.concat(
        all_trades,
        ignore_index=True,
    )

    combined["entry_time"] = pd.to_datetime(
        combined["entry_time"],
        utc=True,
    )

    combined["year"] = (
        combined["entry_time"]
        .dt.tz_convert("America/New_York")
        .dt.year
    )

    if not combined["year"].between(
        START_YEAR, END_YEAR
    ).all():
        raise RuntimeError("Invalid trade years")

    # Overall summary.
    summary = []

    for name, group in combined.groupby("strategy"):
        summary.append({
            "strategy": name,
            **calculate_comparison_metrics(group),
        })

    summary_df = pd.DataFrame(summary)

    # Annual summary.
    annual = []

    for (name, year), group in combined.groupby(
        ["strategy", "year"]
    ):
        annual.append({
            "strategy": name,
            "year": year,
            **calculate_comparison_metrics(group),
        })

    annual_df = pd.DataFrame(annual)

    # Save.
    summary_df.to_csv(
        OUTPUT / "strategy_comparison_summary.csv",
        index=False,
    )

    annual_df.to_csv(
        OUTPUT / "strategy_comparison_annual.csv",
        index=False,
    )

    combined.to_csv(
        OUTPUT / "strategy_comparison_trades.csv",
        index=False,
    )

    # Display results.
    print("\n========== OVERALL COMPARISON ==========")
    print(summary_df.to_string(index=False))

    print("\n========== ANNUAL TOTAL R ==========")
    print(
        annual_df.pivot(
            index="year",
            columns="strategy",
            values="total_r",
        ).round(2).to_string()
    )

    print("\n========== ANNUAL AVG R ==========")
    print(
        annual_df.pivot(
            index="year",
            columns="strategy",
            values="avg_r",
        ).round(4).to_string()
    )

    return summary_df, annual_df, combined


if __name__ == "__main__":
    run_comparison()
