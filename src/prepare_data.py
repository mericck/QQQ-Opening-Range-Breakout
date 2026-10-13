print("PREPARE_DATA STARTED", flush=True)

from pathlib import Path
from datetime import time

import pandas as pd
import pandas_market_calendars as mcal

# Configuration

SYMBOL = "QQQ"
MARKET_TIMEZONE = "America/New_York"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "QQQ_5min_2017_2026.parquet"
)

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"/"tables"

PROCESSED_FILE=PROCESSED_DIR / "QQQ_5min_2017_2026_processed.parquet"

QUALITY_REPORT_FILE = RESULTS_DIR / "data_quality_report.csv"


# Load raw data

print("Loading raw data...")

df = pd.read_parquet(RAW_FILE)

print(f"Raw rows: {len(df):,}")

# Basic schema validation

required_columns = {
    "symbol",
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# Timestamp normalization

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    utc=True
)

df["timestamp"] = df["timestamp"].dt.tz_convert(
    MARKET_TIMEZONE
)

df = df.sort_values("timestamp").reset_index(drop=True)

print(
    "Raw period:",
    df["timestamp"].min(),
    "->",
    df["timestamp"].max(),
)

# Basic raw data integrity checks

duplicate_count = df.duplicated(
    subset=["symbol", "timestamp"]
).sum()

print(f"Duplicate symbol/timestamps: {duplicate_count:,}")


invalid_ohlc = (
    (df["high"] < df["low"])
    | (df["high"] < df["open"])
    | (df["high"] < df["close"])
    | (df["low"] > df["open"])
    | (df["low"] > df["close"])
)

invalid_ohlc_count = invalid_ohlc.sum()

print(f"Invalid OHLC bars: {invalid_ohlc_count:,}")


missing_price_values = df[
    ["open", "high", "low", "close"]
].isna().any(axis=1)

print(
    "Bars with missing OHLC:",
    f"{missing_price_values.sum():,}"
)

# Trading calendar

calendar = mcal.get_calendar("NASDAQ")

schedule = calendar.schedule(
    start_date="2017-01-01",
    end_date="2026-09-30",
)

print(f"Official trading sessions: {len(schedule):,}")


# Convert calendar timestamps to New York time

schedule["market_open"] = (
    schedule["market_open"]
    .dt.tz_convert(MARKET_TIMEZONE)
)

schedule["market_close"] = (
    schedule["market_close"]
    .dt.tz_convert(MARKET_TIMEZONE)
)

# Prepare dateinformation

df["date"] = df["timestamp"].dt.date

schedule = schedule.copy()

schedule["date"] = schedule.index.date

# Filter to official RTH sessions

schedule_lookup = schedule.set_index("date")[
    ["market_open", "market_close"]
]

df = df.join(
    schedule_lookup,
    on="date",
    how="inner",
)


rth = df[
    (df["timestamp"] >= df["market_open"])
    & (df["timestamp"] < df["market_close"])
].copy()


print(f"RTH rows: {len(rth):,}")

# Session level data quality audit

reports = []

for session_date, session in rth.groupby("date"):

    market_open = session["market_open"].iloc[0]
    market_close = session["market_close"].iloc[0]

    expected_timestamps = pd.date_range(
        start=market_open,
        end=market_close - pd.Timedelta(minutes=5),
        freq="5min",
    )

    actual_timestamps = pd.DatetimeIndex(
        session["timestamp"]
    )

    missing_timestamps = expected_timestamps.difference(
        actual_timestamps
    )

    unexpected_timestamps = actual_timestamps.difference(
        expected_timestamps
    )

    # Opening range required by strategy

    expected_opening_range = pd.DatetimeIndex(
        [
            market_open,
            market_open + pd.Timedelta(minutes=5),
            market_open + pd.Timedelta(minutes=10),
        ]
    )

    missing_opening_range = (
        expected_opening_range.difference(
            actual_timestamps
        )
    )

    reports.append(
        {
            "date": session_date,
            "expected_bars": len(expected_timestamps),
            "actual_bars": len(actual_timestamps),
            "missing_bars": len(missing_timestamps),
            "unexpected_bars": len(unexpected_timestamps),
            "opening_range_complete":
                len(missing_opening_range) == 0,
            "market_open": market_open,
            "market_close": market_close,
        }
    )


quality_report = pd.DataFrame(reports)


# Report summary

print("\n========== DATA QUALITY REPORT ==========")

print(
    "Sessions inspected:",
    f"{len(quality_report):,}"
)

print(
    "Sessions with missing bars:",
    f"{(quality_report['missing_bars'] > 0).sum():,}"
)

print(
    "Sessions with unexpected bars:",
    f"{(quality_report['unexpected_bars'] > 0).sum():,}"
)

print(
    "Sessions with incomplete opening range:",
    f"{(~quality_report['opening_range_complete']).sum():,}"
)

print(
    "Total missing RTH bars:",
    f"{quality_report['missing_bars'].sum():,}"
)


# Flag sessions usable for ORB

quality_report["orb_usable"] = (
    (quality_report["missing_bars"] == 0)
    & (quality_report["unexpected_bars"] == 0)
    & (quality_report["opening_range_complete"])
)

usable_dates = set(
    quality_report.loc[
        quality_report["orb_usable"],
        "date",
    ]
)


rth["orb_usable"] = rth["date"].isin(
    usable_dates
)
rth = rth[
    rth["orb_usable"]
].copy()
# Final checks

final_duplicate_count = rth.duplicated(
    subset=["symbol", "timestamp"]
).sum()

if final_duplicate_count > 0:
    raise ValueError(
        f"Processed data contains "
        f"{final_duplicate_count} duplicate bars."
    )


final_invalid_ohlc = (
    (rth["high"] < rth["low"])
    | (rth["high"] < rth["open"])
    | (rth["high"] < rth["close"])
    | (rth["low"] > rth["open"])
    | (rth["low"] > rth["close"])
)

if final_invalid_ohlc.any():
    raise ValueError(
        "Processed data contains invalid OHLC bars."
    )


# Remove helper columns

rth = rth.drop(
    columns=["market_open", "market_close"]
)

# Save outputs

rth.to_parquet(
    PROCESSED_FILE,
    index=False,
)

quality_report.to_csv(
    QUALITY_REPORT_FILE,
    index=False,
)

print(
    "ORB-usable sessions:",
    f"{quality_report['orb_usable'].sum():,}"
)

print(
    "Excluded sessions:",
    f"{(~quality_report['orb_usable']).sum():,}"
)

print("\nProcessed dataset saved to:")
print(PROCESSED_FILE)

print("\nQuality report saved to:")
print(QUALITY_REPORT_FILE)

print("\nData preparation complete.")