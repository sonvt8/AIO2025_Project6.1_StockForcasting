"""
Data Fetcher Service
Fetches real-time FPT stock data from internet
Only fetches missing data from the last date in training dataset
"""

import pandas as pd

from app.config import DATA_FILE

try:
    from vnstock import stock_historical_data
except ImportError:
    stock_historical_data = None


def get_last_date_from_dataset() -> pd.Timestamp | None:
    """
    Get the last date from the training dataset

    Returns:
        Last date in dataset or None if dataset doesn't exist
    """
    if not DATA_FILE.exists():
        return None

    try:
        df = pd.read_csv(DATA_FILE)
        df["time"] = pd.to_datetime(df["time"])
        last_date = df["time"].max()
        return pd.Timestamp(last_date)
    except Exception:
        return None


def fetch_new_data_only(start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    """
    Fetch FPT stock data from internet for a specific date range

    Args:
        start_date: Start date (exclusive, will fetch from next day)
        end_date: End date (inclusive)

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, symbol
    """
    if stock_historical_data is None:
        raise ImportError("vnstock is not installed. Please install it with: pip install vnstock")

    # Start from next business day after start_date
    fetch_start = start_date + pd.offsets.BDay(1)

    # If fetch_start is after end_date, no new data needed
    if fetch_start > end_date:
        return pd.DataFrame(columns=["time", "open", "high", "low", "close", "volume", "symbol"])

    try:
        # Fetch data using vnstock
        try:
            df = stock_historical_data(
                symbol="FPT",
                start_date=fetch_start.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d"),
                resolution="1D",
                type="stock",
            )
        except TypeError:
            try:
                df = stock_historical_data(
                    symbol="FPT",
                    start_date=fetch_start.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d"),
                )
            except Exception:
                df = stock_historical_data(
                    "FPT", fetch_start.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")
                )

        if df is None or df.empty:
            return pd.DataFrame(
                columns=["time", "open", "high", "low", "close", "volume", "symbol"]
            )

        # Normalize column names
        column_mapping = {}

        # Map time/date column
        if "time" in df.columns:
            column_mapping["time"] = "time"
        elif "date" in df.columns:
            column_mapping["date"] = "time"
        elif "ngay" in df.columns:
            column_mapping["ngay"] = "time"
        elif any("time" in col.lower() for col in df.columns):
            time_col = [col for col in df.columns if "time" in col.lower()][0]
            column_mapping[time_col] = "time"
        else:
            first_col = df.columns[0]
            if pd.api.types.is_datetime64_any_dtype(df[first_col]):
                column_mapping[first_col] = "time"
            else:
                raise ValueError("Could not find time/date column in fetched data")

        # Map OHLCV columns
        for target, source_variants in [
            ("open", ["open", "o"]),
            ("high", ["high", "h"]),
            ("low", ["low", "l"]),
            ("close", ["close", "c"]),
            ("volume", ["volume", "vol", "v"]),
        ]:
            found = False
            for variant in source_variants:
                for col in df.columns:
                    if col.lower() == variant.lower():
                        column_mapping[col] = target
                        found = True
                        break
                if found:
                    break
            if not found:
                raise ValueError(f"Could not find {target} column in fetched data")

        # Rename columns
        df = df.rename(columns=column_mapping)

        # Ensure we have required columns
        required_cols = ["time", "open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Convert time to datetime
        df["time"] = pd.to_datetime(df["time"])

        # Add symbol column
        df["symbol"] = "FPT"

        # Select and reorder columns
        df = df[["time", "open", "high", "low", "close", "volume", "symbol"]]

        # Sort by time
        df = df.sort_values("time").reset_index(drop=True)

        # Remove duplicates
        df = df.drop_duplicates(subset=["time"]).reset_index(drop=True)

        return df

    except Exception as e:
        raise RuntimeError(f"Error fetching FPT data: {str(e)}") from e


def fetch_fpt_data(days: int = 120) -> pd.DataFrame:
    """
    Fetch FPT stock data from internet, merging with existing dataset

    Strategy:
    1. Load existing dataset (FPT_train.csv) to get last date
    2. Only fetch new data from last date to today
    3. Merge and return complete dataset

    IMPORTANT: This function ALWAYS returns ALL data from FPT_train.csv + newly fetched data.
    The 'days' parameter is only used when no existing dataset exists (for backward compatibility).

    Args:
        days: Minimum number of days needed (for backward compatibility only,
              used only when no existing dataset exists)

    Returns:
        DataFrame with columns: time, open, high, low, close, volume, symbol
        Format matches FPT_train.csv structure
        Contains ALL data from FPT_train.csv (from 2020-08-03) + newly fetched data
    """
    # Load existing dataset (FPT_train.csv)
    existing_df = None
    last_date = None

    if DATA_FILE.exists():
        try:
            existing_df = pd.read_csv(DATA_FILE)
            existing_df["time"] = pd.to_datetime(existing_df["time"])
            existing_df = existing_df.sort_values("time").reset_index(drop=True)
            last_date = existing_df["time"].max()

            # Debug: log existing dataset info
            first_date = existing_df["time"].min()
            print(
                f"[DEBUG] Loaded existing dataset: {len(existing_df)} records "
                f"from {first_date.date()} to {last_date.date()}"
            )
        except Exception as e:
            # If can't read dataset, continue without it
            print(f"[WARNING] Could not read existing dataset: {e}")
            existing_df = None
            last_date = None

    # Get current date (end of today)
    end_date = pd.Timestamp.today()

    # If we have existing data and last_date is up to date, return ALL existing data
    # This includes ALL historical data from FPT_train.csv (from 2020-08-03)
    if existing_df is not None and last_date is not None:
        # Check if last_date is recent (within last 2 business days)
        days_since_last = (end_date - last_date).days
        if days_since_last <= 2:
            # Data is up to date, return ALL existing data (not just recent data)
            print(
                f"[DEBUG] Data is up to date (last update: {days_since_last} days ago). "
                f"Returning all {len(existing_df)} records from dataset."
            )
            return existing_df

    # Need to fetch new data
    if last_date is None:
        # No existing dataset, fetch minimum required days
        start_date = end_date - pd.Timedelta(days=days * 2)
        new_data = fetch_new_data_only(start_date, end_date)

        if len(new_data) < 20:
            raise ValueError(
                f"Not enough data fetched. Got {len(new_data)} days, need at least 20 days."
            )

        print(
            f"[DEBUG] No existing dataset. Fetched {len(new_data)} records "
            f"from {new_data['time'].min().date()} to {new_data['time'].max().date()}"
        )
        return new_data
    else:
        # Fetch only new data from last_date to today
        # This ensures we get ALL existing data + only the new data
        print(
            f"[DEBUG] Fetching new data from {last_date.date()} to {end_date.date()} "
            f"(existing dataset has {len(existing_df)} records)"
        )
        new_data = fetch_new_data_only(last_date, end_date)

        if len(new_data) == 0:
            # No new data, return ALL existing data (not just recent data)
            print(
                f"[DEBUG] No new data found. Returning all {len(existing_df)} records from dataset."
            )
            return existing_df

        # Merge new data with existing
        # This combines ALL data from FPT_train.csv + newly fetched data
        print(
            f"[DEBUG] Merging {len(existing_df)} existing records with {len(new_data)} new records"
        )
        combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        combined_df = combined_df.sort_values("time").reset_index(drop=True)

        # Remove any duplicates (in case of overlap)
        # Keep the last occurrence (newer data takes precedence)
        before_dedup = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=["time"], keep="last").reset_index(
            drop=True
        )
        if before_dedup != len(combined_df):
            print(
                f"[DEBUG] Removed {before_dedup - len(combined_df)} duplicate records "
                f"(kept newer data)"
            )

        # Ensure we have enough data
        if len(combined_df) < 20:
            raise ValueError(
                f"Not enough data after merge. Got {len(combined_df)} days, need at least 20 days."
            )

        # Debug: log final merged data range
        first_date_final = combined_df["time"].min()
        last_date_final = combined_df["time"].max()
        print(
            f"[DEBUG] Final merged dataset: {len(combined_df)} records "
            f"from {first_date_final.date()} to {last_date_final.date()}"
        )

        return combined_df


def fetch_fpt_data_as_dict_list(days: int = 120) -> tuple[list[dict], dict]:
    """
    Fetch FPT stock data and convert to list of dicts (for API)

    Returns both the data and metadata about what was fetched

    Args:
        days: Minimum number of days needed (for backward compatibility)
              NOTE: This parameter is for backward compatibility only.
              The function ALWAYS returns ALL data from FPT_train.csv + newly fetched data,
              regardless of this parameter value.

    Returns:
        Tuple of (data_list, metadata) where:
        - data_list: List of dicts with keys: time, open, high, low, close, volume
                     Contains ALL historical data from FPT_train.csv + newly fetched data
        - metadata: Dict with keys: fetched_new_data, last_date, total_records
    """
    # Get last date before fetching
    last_date_before = get_last_date_from_dataset()

    # Fetch data - this returns ALL data from FPT_train.csv + newly fetched data
    # The 'days' parameter is ignored when existing dataset exists
    df = fetch_fpt_data(days)

    # Get last date after fetching
    last_date_after = df["time"].max() if len(df) > 0 else None
    first_date_after = df["time"].min() if len(df) > 0 else None

    # Determine if new data was fetched
    fetched_new_data = False
    if last_date_before is None:
        fetched_new_data = True
    elif last_date_after is not None and last_date_after > last_date_before:
        fetched_new_data = True

    # Ensure DataFrame is sorted by time before converting
    df = df.sort_values("time").reset_index(drop=True)

    # Debug: log data range
    if len(df) > 0:
        print(
            f"[DEBUG] fetch_fpt_data_as_dict_list: Returning {len(df)} records "
            f"from {first_date_after.date()} to {last_date_after.date()}"
        )

    # Convert to list of dicts, format time as string
    # IMPORTANT: Convert ALL data, not just recent data
    data_list = []
    for _, row in df.iterrows():
        data_list.append(
            {
                "time": row["time"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
            }
        )

    metadata = {
        "fetched_new_data": fetched_new_data,
        "last_date": last_date_after.strftime("%Y-%m-%d") if last_date_after else None,
        "total_records": len(data_list),
        "previous_last_date": last_date_before.strftime("%Y-%m-%d") if last_date_before else None,
        "first_date": first_date_after.strftime("%Y-%m-%d") if first_date_after else None,
    }

    return data_list, metadata
