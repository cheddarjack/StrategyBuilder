import pandas as pd
import numpy as np

# Define file paths
sc_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_L_August_SC.parquet'
python_path = 'C:\\Projects\\StrategyBuilder\\data\\trade_data.parquet'
output_parquet_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_Compare_Exit.parquet'
stats_parquet_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_Stats_Exit.parquet'

# Read the Parquet files into DataFrames
df_sc = pd.read_parquet(sc_path)
df_python = pd.read_parquet(python_path)

# =========================
# Process SC Data
# =========================

# Identify entries and exits in SC data
trades_sc = []
entry_row = None

for idx, row in df_sc.iterrows():
    if pd.isna(row['Price']) and not pd.isna(row['PositionQuantity']):
        # This is an entry
        entry_row = row.copy()
    elif entry_row is not None:
        # This is an exit (the line after entry)
        exit_row = row.copy()
        # Create a trade record
        trade_record = {
            'entry_datetime': pd.to_datetime(entry_row['DateTime'], utc=True),
            'entry_price_sc': entry_row['FillPrice'],
            'exit_datetime_sc': pd.to_datetime(exit_row['DateTime'], utc=True),
            'exit_price_sc': exit_row['FillPrice'],
            'exit_order_type': exit_row['OrderType']
        }
        trades_sc.append(trade_record)
        entry_row = None  # Reset for the next trade

df_trades_sc = pd.DataFrame(trades_sc)

# Sort SC trades by entry_datetime
df_trades_sc = df_trades_sc.sort_values('entry_datetime').reset_index(drop=True)

# =========================
# Process Python Data
# =========================

# Convert 'datetime' columns to datetime objects with consistent timezone
df_python['datetime'] = pd.to_datetime(df_python['datetime'], utc=True)

# Filter entries and exits
df_entries_python = df_python[df_python['exit_flag'] != True]
df_exits_python = df_python[df_python['exit_flag'] == True]

# Select relevant columns, including bid, ask, and position
df_entries_python = df_entries_python[['trade_id', 'datetime', 'entry_price', 'Bid', 'Ask', 'position']]
df_exits_python = df_exits_python[['trade_id', 'datetime', 'exit_price', 'Bid', 'Ask']]

# Rename columns for clarity
df_entries_python.rename(columns={
    'datetime': 'entry_datetime',
    'Bid': 'entry_bid',
    'Ask': 'entry_ask',
}, inplace=True)

df_exits_python.rename(columns={
    'datetime': 'exit_datetime_python',
    'Bid': 'exit_bid',
    'Ask': 'exit_ask',
}, inplace=True)

# Merge entries and exits on 'trade_id'
df_trades_python = pd.merge(
    df_entries_python,
    df_exits_python,
    on='trade_id',
    how='inner'
)

# Convert datetimes to datetime objects with UTC timezone
df_trades_python['entry_datetime'] = pd.to_datetime(df_trades_python['entry_datetime'], utc=True)
df_trades_python['exit_datetime_python'] = pd.to_datetime(df_trades_python['exit_datetime_python'], utc=True)

# Sort Python trades by entry_datetime
df_trades_python = df_trades_python.sort_values('entry_datetime').reset_index(drop=True)

# =========================
# Combine SC and Python Trades
# =========================

# Since we need to match trades based on entry_datetime with some tolerance, we can use merge_asof

# First, make sure both DataFrames are sorted by entry_datetime
df_trades_sc = df_trades_sc.sort_values('entry_datetime')
df_trades_python = df_trades_python.sort_values('entry_datetime')

# Set the tolerance to an appropriate value, e.g., 1 minute
tolerance = pd.Timedelta('1ms')

# Round datetime to the 3rd milisecond to ensure consistency
df_trades_python['entry_datetime'] = df_trades_python['entry_datetime'].dt.round('3ms')
df_trades_sc['entry_datetime'] = df_trades_sc['entry_datetime'].dt.round('3ms')

combined_trades = pd.merge(df_trades_python, df_trades_sc, on='entry_datetime', how='outer', suffixes=('_sc', '_python'))

# Remove any unwanted columns if they exist
columns_to_remove = ['cumulative_PL', 'profit_loss', 'exit_side', 'entry_side']
combined_trades.drop(columns=[col for col in columns_to_remove if col in combined_trades.columns], inplace=True, errors='ignore')

#remove lines where null values exist
combined_trades = combined_trades.dropna(subset=['entry_price_sc', 'entry_price'])

# Reorder columns
combined_trades = combined_trades[[
    'entry_datetime', 'entry_price_sc', 'entry_price', 
    'position','exit_datetime_sc', 'exit_price_sc', 'exit_order_type',
    'exit_datetime_python', 'exit_price', 'exit_bid', 'exit_ask'
]]

# Rename 'entry_price' and 'exit_price' from Python data for clarity
combined_trades.rename(columns={
    'entry_price': 'entry_price_python',
    'exit_price': 'exit_price_python'
}, inplace=True)

# Define conditions for WL_Python
conditions_python = [
    (combined_trades['position'] == 1) & (combined_trades['entry_price_python'] > combined_trades['exit_price_python']),
    (combined_trades['position'] == 1) & (combined_trades['entry_price_python'] < combined_trades['exit_price_python']),
    (combined_trades['position'] == -1) & (combined_trades['entry_price_python'] > combined_trades['exit_price_python']),
    (combined_trades['position'] == -1) & (combined_trades['entry_price_python'] < combined_trades['exit_price_python'])
]

# Define corresponding choices for WL_Python
choices_python = ['Loss', 'Profit', 'Profit', 'Loss']

# Apply conditions to create WL_Python column
combined_trades['WL_Python'] = np.select(conditions_python, choices_python, default='No Change')

# Define conditions for WL_SC
conditions_sc = [
    (combined_trades['position'] == 1) & (combined_trades['entry_price_sc'] > combined_trades['exit_price_sc']),
    (combined_trades['position'] == 1) & (combined_trades['entry_price_sc'] < combined_trades['exit_price_sc']),
    (combined_trades['position'] == -1) & (combined_trades['entry_price_sc'] > combined_trades['exit_price_sc']),
    (combined_trades['position'] == -1) & (combined_trades['entry_price_sc'] < combined_trades['exit_price_sc'])
]

# Define corresponding choices for WL_SC
choices_sc = ['Loss', 'Profit', 'Profit', 'Loss']

# Apply conditions to create WL_SC column
combined_trades['WL_SC'] = np.select(conditions_sc, choices_sc, default='No Change')



combined_trades['Price_match'] = combined_trades['exit_price_sc'] == combined_trades['exit_price_python']
# Create a column to check if the exit datetimes match with a tollerance of 1ms
combined_trades['time_match'] = abs(combined_trades['exit_datetime_sc'] - combined_trades['exit_datetime_python']) < pd.Timedelta('1ms')

combined_trades['match_bid'] = combined_trades['exit_bid'] == combined_trades['exit_price_sc']
combined_trades['match_ask'] = combined_trades['exit_ask'] == combined_trades['exit_price_sc']
        

# Save the combined DataFrame to a Parquet file
combined_trades.to_parquet(output_parquet_path, index=False)
print(f"Combined trade data saved to {output_parquet_path}")

# =========================
# Generate Trade Statistics
# =========================

stats = {}

# Total number of trades in SC and Python data
stats['Total Trades in SC Data'] = len(df_trades_sc)
stats['Total Trades in Python Data'] = len(df_trades_python)
stats['Total Trades Combined'] = len(combined_trades)

# Percent of price matches based off of 'Price_match' column
stats['Percent Price Matches'] = (combined_trades['Price_match'].sum() / len(combined_trades)) * 100

# Percent of time matches based off of 'time_match' column
stats['Percent Time Matches'] = (combined_trades['time_match'].sum() / len(combined_trades)) * 100

# Percent of trades that are profitable in SC data
stats['Percent Profitable Trades in SC'] = (df_trades_sc['exit_price_sc'] > df_trades_sc['entry_price_sc']).mean() * 100
# Percent of trades that are profitable in Python data
stats['Percent Profitable Trades in Python'] = (df_trades_python['exit_price'] > df_trades_python['entry_price']).mean() * 100

# Convert stats to DataFrame and save
stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])
stats_df.to_parquet(stats_parquet_path, index=False)
print(f"Trade statistics saved to {stats_parquet_path}")
