"""
Test script for data fetcher service
Run: python test_data_fetcher.py
"""

from app.services.data_fetcher import (
    fetch_fpt_data,
    fetch_fpt_data_as_dict_list,
    get_last_date_from_dataset,
)


def test_get_last_date():
    """Test getting last date from dataset"""
    print("=" * 60)
    print("TEST 1: Get Last Date from Dataset")
    print("=" * 60)
    last_date = get_last_date_from_dataset()
    if last_date:
        print(f"✅ Last date in dataset: {last_date.strftime('%Y-%m-%d')}")
    else:
        print("⚠️  No dataset found or error reading dataset")
    print()


def test_fetch_data():
    """Test fetching data"""
    print("=" * 60)
    print("TEST 2: Fetch Data (Smart Fetch)")
    print("=" * 60)
    try:
        df = fetch_fpt_data(days=120)
        print(f"✅ Total records fetched: {len(df)}")
        min_date = df["time"].min().strftime("%Y-%m-%d")
        max_date = df["time"].max().strftime("%Y-%m-%d")
        print(f"✅ Date range: {min_date} to {max_date}")
        print("\nFirst 3 rows:")
        print(df[["time", "close", "volume"]].head(3).to_string(index=False))
        print("\nLast 3 rows:")
        print(df[["time", "close", "volume"]].tail(3).to_string(index=False))
    except Exception as e:
        print(f"❌ Error: {e}")
    print()


def test_fetch_as_dict_list():
    """Test fetching as dict list with metadata"""
    print("=" * 60)
    print("TEST 3: Fetch as Dict List (with Metadata)")
    print("=" * 60)
    try:
        data_list, metadata = fetch_fpt_data_as_dict_list(days=120)
        print(f"✅ Total records: {metadata['total_records']}")
        print(f"✅ Latest date: {metadata['last_date']}")
        print(f"✅ Previous last date: {metadata.get('previous_last_date', 'N/A')}")
        print(f"✅ Fetched new data: {metadata['fetched_new_data']}")

        if metadata["fetched_new_data"]:
            print("\n🔄 New data was fetched from internet")
        else:
            print("\nℹ️  Using existing dataset (no new data needed)")

        print("\nFirst record:")
        print(f"  Date: {data_list[0]['time']}")
        print(f"  Close: {data_list[0]['close']}")
        print(f"  Volume: {data_list[0]['volume']}")

        print("\nLast record:")
        print(f"  Date: {data_list[-1]['time']}")
        print(f"  Close: {data_list[-1]['close']}")
        print(f"  Volume: {data_list[-1]['volume']}")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
    print()


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("DATA FETCHER TEST SUITE")
    print("=" * 60 + "\n")

    test_get_last_date()
    test_fetch_data()
    test_fetch_as_dict_list()

    print("=" * 60)
    print("TESTING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
