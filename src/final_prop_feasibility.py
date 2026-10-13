
"""
Final QQQ ORB prop-firm feasibility analysis.

Compare:
- 30, 60, 90 trade-opportunity horizons
- IID bootstrap vs moving block bootstrap
- Static +$2500 / -$1500 barriers
- Fixed risk per trade
- 4-cent round-trip transaction cost

This is a scenario study, not a live execution model.
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
HORIZONS = [30, 60, 90]

ROUND_TRIP_COST = 0.04

N_SIMULATIONS = 20_000
BLOCK_LENGTH = 10
SEED = 42


def load_net_r():
    df = pd.read_csv(INPUT)
    df = df[df["strategy"] == "primary"].copy()

    required = [
        "strategy", "date", "entry_time",
        "r_multiple", "initial_risk"
    ]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if df.empty:
        raise ValueError("No primary trades found")

    if df["date"].duplicated().any():
        raise ValueError("Multiple trades on same date")

    df["entry_time"] = pd.to_datetime(
        df["entry_time"], utc=True
    )
    df = df.sort_values("entry_time")

    gross = df["r_multiple"].to_numpy(dtype=float)
    stop = df["initial_risk"].to_numpy(dtype=float)

    if not np.isfinite(gross).all():
        raise ValueError("Invalid R multiples")

    if not np.isfinite(stop).all() or (stop <= 0).any():
        raise ValueError("Invalid stop distances")

    net = gross - ROUND_TRIP_COST / stop

    print(f"Primary OOS trades: {len(net)}")
    print(f"Net expectancy: {net.mean():+.5f}R")

    return net


def generate_paths(net_r, method, rng):
    """
    Shape: (N_SIMULATIONS, max(HORIZONS))

    IID: independent historical trade draws.
    Block: concatenate randomly selected consecutive
    blocks of historical trade outcomes.
    """
    n = len(net_r)
    horizon = max(HORIZONS)

    if method == "iid":
        indices = rng.integers(
            0, n, size=(N_SIMULATIONS, horizon)
        )

    elif method == "block":
        if BLOCK_LENGTH > n:
            raise ValueError("Block exceeds sample length")

        blocks_needed = int(
            np.ceil(horizon / BLOCK_LENGTH)
        )

        starts = rng.integers(
            0,
            n - BLOCK_LENGTH + 1,
            size=(N_SIMULATIONS, blocks_needed),
        )

        offsets = np.arange(BLOCK_LENGTH)

        indices = (
            starts[:, :, None]
            + offsets[None, None, :]
        ).reshape(N_SIMULATIONS, -1)[:, :horizon]

    else:
        raise ValueError(f"Unknown method: {method}")

    return net_r[indices]


def evaluate_paths(r_paths, risk_per_trade):
    """
    Record the FIRST barrier reached on each path.

    Status:
      1 = pass
     -1 = fail
      0 = neither barrier reached
    """
    pnl = np.cumsum(
        r_paths * risk_per_trade,
        axis=1,
    )

    upper_hit = pnl >= PROFIT_TARGET
    lower_hit = pnl <= -MAX_LOSS

    any_hit = upper_hit | lower_hit
    has_hit = any_hit.any(axis=1)

    first_index = any_hit.argmax(axis=1)

    first_status = np.zeros(
        len(r_paths), dtype=np.int8
    )

    rows = np.flatnonzero(has_hit)
    cols = first_index[rows]

    first_status[rows] = np.where(
        upper_hit[rows, cols], 1, -1
    )

    # Trade count starts at 1.
    first_trade = np.where(
        has_hit, first_index + 1, np.inf
    )

    return first_status, first_trade


def summarize(
    method, risk, horizon, status, first_trade
):
    reached = first_trade <= horizon

    passed = reached & (status == 1)
    failed = reached & (status == -1)
    timeout = ~reached

    p_pass = passed.mean()
    p_fail = failed.mean()
    p_timeout = timeout.mean()

    # Binomial Monte Carlo uncertainty ONLY.
    mc_se = np.sqrt(
        p_pass * (1 - p_pass) / len(status)
    )

    pass_times = first_trade[passed]

    return {
        "method": method,
        "risk_per_trade": risk,
        "horizon_trades": horizon,
        "pass_probability": p_pass,
        "fail_probability": p_fail,
        "timeout_probability": p_timeout,
        "mc_se_pass": mc_se,
        "mc_margin_95": 1.96 * mc_se,
        "median_trades_to_pass": (
            np.median(pass_times)
            if len(pass_times) else np.nan
        ),
        "n_pass": int(passed.sum()),
        "n_fail": int(failed.sum()),
        "n_timeout": int(timeout.sum()),
    }


def run():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    net_r = load_net_r()

    records = []

    # Each method uses the same R paths for all risk
    # levels and all time horizons.
    for method_index, method in enumerate(
        ["iid", "block"]
    ):
        rng = np.random.default_rng(
            SEED + method_index
        )

        r_paths = generate_paths(
            net_r, method, rng
        )

        for risk in RISK_LEVELS:
            status, first_trade = evaluate_paths(
                r_paths, risk
            )

            for horizon in HORIZONS:
                result = summarize(
                    method,
                    risk,
                    horizon,
                    status,
                    first_trade,
                )

                total = (
                    result["pass_probability"]
                    + result["fail_probability"]
                    + result["timeout_probability"]
                )
                assert np.isclose(total, 1.0)

                records.append(result)

    results = pd.DataFrame(records)

    results.to_csv(
        OUTPUT / "final_prop_feasibility.csv",
        index=False,
    )

    for method in ["iid", "block"]:
        subset = results[
            results["method"] == method
        ]

        print(f"\n========== {method.upper()} ==========")

        for metric in [
            "pass_probability",
            "fail_probability",
            "timeout_probability",
        ]:
            table = subset.pivot(
                index="risk_per_trade",
                columns="horizon_trades",
                values=metric,
            )

            print(f"\n{metric.upper()} (%)")
            print((table * 100).round(2).to_string())

    print("\nSaved: results/tables/final_prop_feasibility.csv")

    return results


if __name__ == "__main__":
    run()
