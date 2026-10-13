
"""
Robustness analysis for QQQ ORB primary strategy.

Methods:
- Transaction cost sensitivity
- Moving block bootstrap
- Confidence intervals for mean R
- Bootstrap drawdown distribution

No strategy parameters or execution rules are changed.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/tables/final_oos_trades.csv"
OUTPUT = ROOT / "results/tables"

STRATEGY = "primary"

COSTS = [0.00, 0.02, 0.04, 0.06, 0.10]

BLOCK_LENGTHS = [5, 10, 20]

N_BOOTSTRAPS = 5000
RANDOM_SEED = 42


# ============================================================
# 1. LOAD AND VALIDATE TRADES
# ============================================================

def load_trades():

    df = pd.read_csv(INPUT)

    df = df[df["strategy"] == STRATEGY].copy()

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True
    )

    df = df.sort_values("entry_time").reset_index(
        drop=True
    )

    if df.empty:
        raise ValueError("No primary trades found")

    if df["date"].duplicated().any():
        raise ValueError("Multiple trades on the same date")

    if (df["initial_risk"] <= 0).any():
        raise ValueError("Invalid initial risk")

    if df[["r_multiple", "initial_risk"]].isna().any().any():
        raise ValueError("Missing trade values")

    if not np.isfinite(
        df[["r_multiple", "initial_risk"]].to_numpy()
    ).all():
        raise ValueError("Non-finite trade values")

    return df


# ============================================================
# 2. TRANSACTION COST MODEL
# ============================================================

def apply_costs(trades, round_trip_cost):

    df = trades.copy()

    df["cost_r"] = (
        round_trip_cost / df["initial_risk"]
    )

    df["net_r"] = (
        df["r_multiple"] - df["cost_r"]
    )

    return df


# ============================================================
# 3. PERFORMANCE STATISTICS
# ============================================================

def max_drawdown(r_values):

    equity = np.concatenate([
        [0.0],
        np.cumsum(r_values),
    ])

    running_peak = np.maximum.accumulate(equity)

    drawdown = running_peak - equity

    return float(drawdown.max())


def performance_metrics(r_values):

    r = np.asarray(r_values, dtype=float)

    return {
        "trades": len(r),
        "avg_r": float(np.mean(r)),
        "total_r": float(np.sum(r)),
        "std_r": float(np.std(r, ddof=1)),
        "standard_error_iid": float(
            np.std(r, ddof=1) / np.sqrt(len(r))
        ),
        "max_drawdown_r": max_drawdown(r),
    }


# ============================================================
# 4. MOVING BLOCK BOOTSTRAP
# ============================================================

def moving_block_bootstrap(
    r_values,
    block_length,
    n_bootstraps,
    seed,
):
    """
    Resample consecutive trading observations.

    The original number of observations is preserved.
    """

    r = np.asarray(r_values, dtype=float)

    n = len(r)

    if block_length > n:
        raise ValueError("Block longer than sample")

    rng = np.random.default_rng(seed)

    bootstrap_means = np.empty(n_bootstraps)
    bootstrap_drawdowns = np.empty(n_bootstraps)

    possible_starts = n - block_length + 1

    blocks_needed = int(np.ceil(n / block_length))

    for b in range(n_bootstraps):

        starts = rng.integers(
            low=0,
            high=possible_starts,
            size=blocks_needed,
        )

        indices = np.concatenate([
            np.arange(start, start + block_length)
            for start in starts
        ])[:n]

        sample = r[indices]

        bootstrap_means[b] = sample.mean()

        bootstrap_drawdowns[b] = max_drawdown(sample)

    return bootstrap_means, bootstrap_drawdowns


# ============================================================
# 5. BOOTSTRAP SUMMARY
# ============================================================

def bootstrap_statistics(means, drawdowns):

    return {
        "ci_lower_95": np.percentile(means, 2.5),
        "ci_upper_95": np.percentile(means, 97.5),
        "bootstrap_se": np.std(means, ddof=1),
        "fraction_means_above_zero": np.mean(means > 0),
        "median_max_dd": np.median(drawdowns),
        "p95_max_dd": np.percentile(drawdowns, 95),
    }


# ============================================================
# 6. MAIN ANALYSIS
# ============================================================

def run():

    trades = load_trades()

    print(f"Loaded {len(trades)} primary trades")

    cost_results = []
    bootstrap_results = []

    for cost in COSTS:

        net_trades = apply_costs(
            trades,
            round_trip_cost=cost,
        )

        net_r = net_trades["net_r"].to_numpy()

        perf = performance_metrics(net_r)

        cost_results.append({
            "round_trip_cost": cost,
            **perf,
        })

        for block_length in BLOCK_LENGTHS:

            means, drawdowns = moving_block_bootstrap(
                r_values=net_r,
                block_length=block_length,
                n_bootstraps=N_BOOTSTRAPS,
                seed=RANDOM_SEED,
            )

            stats = bootstrap_statistics(
                means,
                drawdowns,
            )

            bootstrap_results.append({
                "round_trip_cost": cost,
                "block_length": block_length,
                "observed_avg_r": perf["avg_r"],
                **stats,
            })

    cost_df = pd.DataFrame(cost_results)

    bootstrap_df = pd.DataFrame(bootstrap_results)

    OUTPUT.mkdir(parents=True, exist_ok=True)

    cost_df.to_csv(
        OUTPUT / "robustness_costs.csv",
        index=False,
    )

    bootstrap_df.to_csv(
        OUTPUT / "robustness_bootstrap.csv",
        index=False,
    )

    print("\n========== COST SENSITIVITY ==========")

    print(
        cost_df[
            [
                "round_trip_cost",
                "avg_r",
                "total_r",
                "max_drawdown_r",
            ]
        ].round(4).to_string(index=False)
    )

    print("\n========== BOOTSTRAP RESULTS ==========")

    print(
        bootstrap_df[
            [
                "round_trip_cost",
                "block_length",
                "observed_avg_r",
                "ci_lower_95",
                "ci_upper_95",
                "p95_max_dd",
            ]
        ].round(4).to_string(index=False)
    )

    return cost_df, bootstrap_df


if __name__ == "__main__":
    run()
