import yaml
import sys
import time
import pandas as pd

# Update your sys.path to include the project directory
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

if __name__ == "__main__":
    start_time = time.time()
    print('Program started')

    # Load configuration 
    config = load_config('/home/cheddarjackk/StrategyBuilder/config/config.yaml')
    
    # Load tick data with specified data types
    print('Loading tick data')
    tick_data = load_data(config['data']['preprocessed_data_path'])

    # Load trades data if needed (ensure data types are set appropriately)
    # trades_data = load_data(config['output']['trade_data_path'])

    # Process tick data
    print('Starting data processing')
    process_start_time = time.time()
    tick_data = process_tick_data(tick_data, config['parameters'])
    process_end_time = time.time()
    process_elapsed_time = process_end_time - process_start_time
    print(f'Data processing completed. Elapsed time: {process_elapsed_time:.2f} seconds')

    # Generate trading signals
    print('Generating trading signals')
    signal_time = time.time()
    tick_data = generate_signals(tick_data, config['parameters'])
    signal_elapsed_time = time.time() - signal_time
    print(f'Signal generation completed. Elapsed time: {signal_elapsed_time:.2f} seconds')

    # Simulate trades
    print('Simulating trades')
    trade_simulate_time = time.time()
    tick_data = simulate_trades(tick_data, config['parameters'])
    trade_simulate_elapsed_time = time.time() - trade_simulate_time
    print(f'Trade simulation completed. Elapsed time: {trade_simulate_elapsed_time:.2f} seconds')

    # Extract trades
    print('Extracting trades')
    extraction_time = time.time()
    trades_data = extract_trades(tick_data)
    extraction_elapsed_time = time.time() - extraction_time
    print(f'Trade extraction completed. Elapsed time: {extraction_elapsed_time:.2f} seconds')

    # Save processed tick data as a Parquet file
    print('Saving processed tick data to Parquet file')
    save_data(tick_data, config['output']['processed_tick_data_path'])

    # Save trades data to Parquet file
    print('Saving trades data to Parquet file')
    save_data(trades_data, config['output']['trade_data_path'])

    # Calculate metrics
    print('Calculating metrics')
    metrics_time = time.time()
    metrics, time_metrics = calculate_trade_metrics(trades_data, config['parameters'])
    metrics_elapsed_time = time.time() - metrics_time
    print(f'Metrics calculation completed. Elapsed time: {metrics_elapsed_time:.2f} seconds')

    # Convert metrics dictionary to DataFrames
    metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
    columns = ['Metric', 'total', '22:00', '23:00', '0:00', '1:00', '2:00', '3:00', '4:00', 
               '5:00', '6:00', '7:00', '8:00', '9:00', '10:00', '11:00', '12:00', '13:00', 
               '14:00', '15:00', '16:00', '17:00', '18:00', '19:00', '20:00', '21:00']
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