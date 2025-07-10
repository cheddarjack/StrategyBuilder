import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import yaml
import time
import os
import shutil

def clear_marked_dir(dir_path):
    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def main():
        
    start_time = time.time()
    print("Loading parquet file...")
    # Read the initial parquet file

    def load_config(config_path):
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

    config = load_config('/home/cheddarjackk/Developer/StrategyBuilder/c__config/config.yaml')

    table = pq.read_table(config['data']['snipped_data_path'])
    df = table.to_pandas()

    print("Data loaded successfully. Beginning mundane processes...")

    # Strip leading/trailing spaces from column names
    df.columns = df.columns.str.strip()

    # Remove unwanted columns
    df.drop(columns=['Open'], inplace=True)

    # Rename 'High' to 'Ask' and 'Low' to 'Bid'
    df.rename(columns={'High': 'Ask', 'Low': 'Bid'}, inplace=True)

    # Standardize Time values to ensure they all have milliseconds
    df['Time'] = df['Time'].apply(lambda x: x if '.' in x else x + '.000')

    # Combine Date and Time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'].astype(str),
                                    format='%Y/%m/%d %H:%M:%S.%f', errors='coerce')

    # Drop the original 'Time' and 'Date' columns
    df.drop(columns=['Time', 'Date'], inplace=True)

    df['unique_id'] = range(len(df))

    # Sort by datetime to ensure proper ordering
    df.sort_values(by='datetime', inplace=True)

    # Reset the index to avoid saving it to the Parquet file
    df.reset_index(drop=True, inplace=True)

    # ----------------- Additional Processing -----------------

    print("Simulating candles...")

    # Create a new DataFrame with all 15-second intervals from the min to max datetime in the original data
    full_range = pd.date_range(start=df['datetime'].min(), end=df['datetime'].max(), freq='15s')
    df_full_range = pd.DataFrame({'datetime': full_range})
    df_full_range['is_new_interval'] = 1

    df['is_new_interval'] = 2

    # Concatenate original data with the full range DataFrame to add missing intervals
    df_combined = pd.concat([df, df_full_range], ignore_index=True)

    # Sort by datetime again to ensure proper ordering
    df_combined.sort_values(by=['datetime', 'is_new_interval', 'unique_id'], inplace=True)

    # Reset index after sorting
    df_combined.reset_index(drop=True, inplace=True)

    # Back-fill relevant columns to fill the missing intervals
    df_combined['Last'] = df_combined['Last'].bfill()

    # Set 'Open' to 'Last' for newly created intervals where 'Volume' is NaN
    mask = df_combined['Volume'].isna()
    df_combined.loc[mask, 'Open'] = df_combined['Last']

    # Forward-fill 'Open' for rows where 'Volume' is not NaN
    df_combined['Open'] = df_combined['Open'].ffill()

    # Create a group identifier for each 15-second interval
    df_combined['group'] = (df_combined['Volume'].isna()).cumsum()

    # Sort by datetime within groups to ensure proper order
    df_combined.sort_values(['group', 'datetime'], inplace=True)

    # Within each group, calculate cumulative high and low
    df_combined['High'] = df_combined.groupby('group')['Last'].cummax()
    df_combined['Low'] = df_combined.groupby('group')['Last'].cummin()

    # Drop the temporary group column
    df_combined.drop(columns=['group'], inplace=True)

    print("Moving decimal points..")

    def move_decimal(value):
        if value > 10000:
            value =  value / 100 if pd.notna(value) else value
        return value

    # Apply the function to the specified columns
    for column in ['Last', 'Low', 'Open', 'High', 'Bid', 'Ask']:
        if column in df_combined.columns:
            df_combined[column] = df_combined[column].apply(move_decimal)

    print("Seperating Datetime...")

    # Add separate datetime column adjusted to Central Time from the data's base UTC time
    df_combined['datetime_central'] = df_combined['datetime'] - pd.Timedelta(hours=5)
    df_combined['date_ct'] = df_combined['datetime_central'].dt.date
    df_combined.drop(columns=['datetime_central'], inplace=True)

    # ----------------- Move session_id Calculation Here -----------------

    print("Calculating session IDs...")

    # Create a boolean mask for rows where Volume is not null
    volume_not_null = df_combined['Volume'].notnull()

    # Compute time differences only for rows where Volume is not null
    time_diff = df_combined.loc[volume_not_null, 'datetime'].diff()

    # Initialize 'time_diff' column with NaT (Not a Time)
    df_combined.loc[volume_not_null, 'time_diff'] = time_diff

    # Assign the computed time differences back to the corresponding rows
    df_combined.loc[volume_not_null, 'time_diff'] = time_diff

    # Define closure threshold
    closure_threshold = pd.Timedelta(hours=1)

    # Determine where the market is considered closed based on 'time_diff'
    df_combined['is_market_closed'] = df_combined['time_diff'] > closure_threshold

    # Fill NaN values in 'is_market_closed' with False (since NaN means no change)
    df_combined['is_market_closed'] = df_combined['is_market_closed'].fillna(False)

    # Compute cumulative sum to get session IDs
    df_combined['session_id'] = df_combined['is_market_closed'].cumsum().astype(int)

    # Optionally, drop the helper columns if no longer needed
    df_combined.drop(columns=['time_diff', 'is_market_closed'], inplace=True)

    print("Print first few lines of the final DataFrame...")
    print(df_combined.head())

    # ----------------- Save the final DataFrame -----------------

    print("Saving final data to parquet file...")

    # Create output directory if it doesn't exist
    output_dir = '/home/cheddarjackk/Developer/VAmodel/data/data_edit/3_day_data_preprocessed_ext'
    clear_marked_dir(output_dir)  # Clear old files
    os.makedirs(output_dir, exist_ok=True)

    # Save each session separately
    sessions = df_combined.groupby('session_id')

    for session_id, session_data in tqdm(sessions, desc='Saving Sessions'):
        start_date = session_data['datetime'].iloc[0].strftime('%Y-%m-%d')
        end_date = session_data['datetime'].iloc[-1].strftime('%Y-%m-%d')
        session_name = f'session_{session_id}_{start_date}_to_{end_date}'

        output_path = os.path.join(output_dir, f'{session_name}.parquet')
        table = pa.Table.from_pandas(session_data, preserve_index=False)
        pq.write_table(table, output_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("Processing complete.")

if __name__ == "__main__":
    main()