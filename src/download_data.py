import os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = RAW_DATA_DIR / "QQQ_5min_2017_2026.parquet"


# Load API credentials
load_dotenv(PROJECT_ROOT / ".env")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")

if not API_KEY or not SECRET_KEY:
    raise ValueError("Alpaca API credentials not found in .env")


# Alpaca client
client = StockHistoricalDataClient(
    API_KEY,
    SECRET_KEY
)


# Historical data request
request_params = StockBarsRequest(
    symbol_or_symbols="QQQ",
    timeframe=TimeFrame(5, TimeFrameUnit.Minute),
    start=datetime(
        2017, 1, 1,
        tzinfo=ZoneInfo("America/New_York")
    ),
    end=datetime(
        2026, 10, 1,
        tzinfo=ZoneInfo("America/New_York")
    )
)


print("Downloading QQQ 5-minute data...")

bars = client.get_stock_bars(request_params)
df = bars.df.reset_index()


# Basic checks
if df.empty:
    raise ValueError("No data returned from Alpaca.")

print(f"Rows downloaded: {len(df):,}")
print(f"First timestamp: {df['timestamp'].min()}")
print(f"Last timestamp:  {df['timestamp'].max()}")


# Save raw data
df.to_parquet(
    OUTPUT_FILE,
    index=False
)

print("\nRaw dataset saved to:")
print(OUTPUT_FILE)

