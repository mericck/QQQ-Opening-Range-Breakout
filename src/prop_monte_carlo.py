
"""
Monte Carlo simulation for QQQ ORB prop-firm challenge.

Model:
- Fixed dollar risk per trade
- Empirical net R distribution
- IID bootstrap sampling
- Static profit and loss barriers
- Finite trade horizon

No daily loss or trailing drawdown rules yet.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

INPUT = ROOT / "results/tables/final_oos_trades.csv"
OUTPUT = ROOT / "results/tables"

PROFIT_TARGET = 2500.0
MAX_LOSS = 1500.0

RISK_LEVELS = [25, 50, 75, 100, 150]

ROUND_TRIP_COST = 0.04

N_SIMULATIONS = 10000
MAX_TRADES = 1000
SEED = 42


# ============================================================
# 1. LOAD EMPIRICAL NET R DISTRIBUTION
# ============================================================

def load_net_returns():

    df = pd.read_csv(INPUT)

    df = df[df["strategy"] == "primary"].copy()

    if df.empty:
        raise ValueError("No primary OOS trades found")

    if df["date"].duplicated().any():
        raise ValueError("Multiple trades on same date")

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True
    )

    df = df.sort_values("entry_time")

    required = ["r_multiple", "initial_risk"]

    if df[required].isna().any().any():
        raise ValueError("Missing trade data")

    if (df["initial_risk"] <= 0).any():
        raise ValueError("Invalid stop distance")

    net_r = (
        df["r_multiple"].to_numpy(dtype=float)
        - ROUND_TRIP_COST
        / df["initial_risk"].to_numpy(dtype=float)
    )

    if not np.isfinite(net_r).all():
        raise ValueError("Invalid net returns")

    print(f"Loaded {len(net_r)} OOS trades")
    print(f"Net Average R: {net_r.mean():.5f}")
    print(f"Net Std R: {net_r.std(ddof=1):.5f}")

    return net_r


# ============================================================
# 2. SIMULATE ONE CHALLENGE
# ============================================================

def simulate_challenge(
    net_returns,
    risk_per_trade,
    rng,
):

    pnl = 0.0
    peak = 0.0
    max_drawdown = 0.0

    for trade_number in range(1, MAX_TRADES + 1):

        # Draw one historical net R outcome.
        r = rng.choice(net_returns)

        # Convert R to dollars.
        trade_pnl = risk_per_trade * r

        pnl += trade_pnl

        peak = max(peak, pnl)

        max_drawdown = max(
            max_drawdown,
            peak - pnl,
        )

        # Absorbing upper barrier.
        if pnl >= PROFIT_TARGET:
            return {
                "outcome": "pass",
                "trades": trade_number,
                "final_pnl": pnl,
                "max_drawdown": max_drawdown,
            }

        # Absorbing lower barrier.
        if pnl <= -MAX_LOSS:
            return {
                "outcome": "fail",
                "trades": trade_number,
                "final_pnl": pnl,
                "max_drawdown": max_drawdown,
            }

    # Neither barrier was reached.
    return {
        "outcome": "unresolved",
        "trades": MAX_TRADES,
        "final_pnl": pnl,
        "max_drawdown": max_drawdown,
    }


# ============================================================
# 3. RUN MANY SIMULATIONS
# ============================================================

def run_monte_carlo(net_returns, risk_per_trade, seed):

    rng = np.random.default_rng(seed)

    results = []

    for _ in range(N_SIMULATIONS):

        result = simulate_challenge(
            net_returns=net_returns,
            risk_per_trade=risk_per_trade,
            rng=rng,
        )

        results.append(result)

    return pd.DataFrame(results)


# ============================================================
# 4. CALCULATE PROBABILITIES
# ============================================================

def summarize(results, risk_per_trade):

    passed = results["outcome"] == "pass"
    failed = results["outcome"] == "fail"
    unresolved = results["outcome"] == "unresolved"

    # Conditional on actually passing.
    pass_trades = results.loc[passed, "trades"]

    # Conditional on reaching either barrier.
    resolved = results.loc[~unresolved]

    return {
        "risk_per_trade": risk_per_trade,
        "pass_probability": passed.mean(),
        "fail_probability": failed.mean(),
        "unresolved_probability": unresolved.mean(),
        "median_trades_to_pass": (
            pass_trades.median()
            if len(pass_trades) else np.nan
        ),
        "mean_trades_to_resolution": (
            resolved["trades"].mean()
            if len(resolved) else np.nan
        ),
        "p95_max_drawdown": results[
            "max_drawdown"
        ].quantile(0.95),
    }


# ============================================================
# 5. MAIN
# ============================================================

def run():

    OUTPUT.mkdir(parents=True, exist_ok=True)

    net_returns = load_net_returns()

    summaries = []
    all_paths = []

    for risk in RISK_LEVELS:

        print(f"Simulating risk ${risk}...", flush=True)

        results = run_monte_carlo(
            net_returns=net_returns,
            risk_per_trade=risk,
            seed=SEED,
        )

        results["risk_per_trade"] = risk

        all_paths.append(results)

        summaries.append(
            summarize(results, risk)
        )

    summary_df = pd.DataFrame(summaries)

    paths_df = pd.concat(
        all_paths,
        ignore_index=True,
    )

    summary_df.to_csv(
        OUTPUT / "prop_mc_summary.csv",
        index=False,
    )

    paths_df.to_csv(
        OUTPUT / "prop_mc_paths.csv",
        index=False,
    )

    print("\n========== MONTE CARLO SUMMARY ==========")

    print(
        summary_df.round(4).to_string(index=False)
    )

    print("\nCheck probability totals:")

    totals = (
        summary_df["pass_probability"]
        + summary_df["fail_probability"]
        + summary_df["unresolved_probability"]
    )

    assert np.allclose(totals, 1.0)

    return summary_df, paths_df


if __name__ == "__main__":
    run()
