import yaml
import sys
import time
import pandas as pd
import os
from glob import glob
import gc
from multiprocessing import Pool, cpu_count

sys.path.append('/home/cheddarjackk/StrategyBuilder')

# Import the functions from the modules
from f__modules.strat import process_tick_data
from f__modules.trades_simulate import generate_signals, simulate_trades, extract_trades
from f__modules.metrics import calculate_trade_metrics
from f__modules.dat_handler import load_data, save_data

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def process_single_session(file_path):
    print(f'Processing file: {file_path}')

    # Load tick data
    tick_data = load_data(file_path)

    # Process tick data
    tick_data = process_tick_data(tick_data, config['parameters'])
    tick_data = generate_signals(tick_data, config['parameters'])
    tick_data = simulate_trades(tick_data, config['parameters'])

    trades_data = extract_trades(tick_data)

    processed_tick_data_path = os.path.join(config['output']['processed_tick_data_dir'], os.path.basename(file_path))
    save_data(tick_data, processed_tick_data_path)

    # Free up memory
    del tick_data
    gc.collect()

    return trades_data

if __name__ == "__main__":
    start_time = time.time()
    print('Program started')

    # Load paths
    config = load_config('/home/cheddarjackk/StrategyBuilder/c__config/config.yaml')
    data_dir = '/home/cheddarjackk/StrategyBuilder/d__data/3_day_data_preprocessed/'

    # List all parquet files in the directory
    all_files = glob(os.path.join(data_dir, 'session_*.parquet'))

    # Define the start and end session IDs
    start_session_id = 170
    end_session_id = 181

    # Filter files based on session IDs in file names
    selected_files = []
    for file in all_files:
        file_name = os.path.basename(file)
        parts = file_name.replace('.parquet', '').split('_')

        try:
            session_id = int(float(parts[1]))
        except (IndexError, ValueError) as e:
            print(f'Filename {file_name} does not match expected pattern or contains invalid session ID. Skipping.')
            continue

        # Check if the session_id is within the desired range
        if start_session_id <= session_id <= end_session_id:
            selected_files.append(file)

    # Sort the files by session ID for consistent processing
    selected_files.sort(key=lambda x: int(float(os.path.basename(x).split('_')[1])))

    # Ensure the config is pickleable (avoid issues with multiprocessing)
    import copy
    config_for_multiprocessing = copy.deepcopy(config)

    num_workers = 8 #cpu_count()  # Use the number of CPU cores available
    print(f'Using {num_workers} worker processes.')

    with Pool(processes=num_workers) as pool:
        # Map the processing function to the list of selected files
        all_trades_data = pool.map(process_single_session, selected_files)

    # Concatenate all trades data into a single DataFrame
    if all_trades_data:
        trades_data_combined = pd.concat(all_trades_data, ignore_index=True)
    else:
        trades_data_combined = pd.DataFrame()   


    # Save the combined trades data
    print('Saving combined trades data to Parquet file')
    save_data(trades_data_combined, config['output']['trade_data_path'])

    # Calculate metrics
    print('Calculating metrics')
    metrics, time_metrics = calculate_trade_metrics(trades_data_combined, config['parameters'])

    # Convert metrics dictionary to DataFrames
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    columns = ['Metric', 'total'] + [f'{hour}:00' for hour in range(22, 24)] + [f'{hour}:00' for hour in range(0, 22)]
    time_metrics_df = pd.DataFrame([
        [key] + value for key, value in time_metrics.items()
    ], columns=columns)

    # Save metrics DataFrames to Parquet files
    print('Saving metrics to Parquet files') 
    metrics_df.to_parquet(config['output']['metrics_data_path'], index=False)
    time_metrics_df.to_parquet(config['output']['time_metrics'], index=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f'Program ended. Total elapsed time: {elapsed_time:.2f} seconds')
