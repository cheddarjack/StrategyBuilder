import pandas as pd
import numpy as np

# Define file paths
sc_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_L_August_SC.parquet'
python_path = 'C:\\Projects\\StrategyBuilder\\data\\trade_data.parquet'
output_parquet_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_Compare.parquet'
stats_parquet_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_Stats.parquet'

# Read the Parquet files into DataFrames
df1 = pd.read_parquet(sc_path)
df2 = pd.read_parquet(python_path)

# Filter out exit lines
df1 = df1[df1['PositionQuantity'].notna()]  # Keep only rows where PositionQuantity is not null
df2 = df2[df2['exit_flag'] != True]  # Keep only rows where exit_flag is not true

# Standardize and process columns
df1 = df1.rename(columns={'DateTime': 'datetime', 'Price': 'price', 'FillPrice': 'fill_price'})
df1['price'] = df1['price'].fillna(df1['fill_price'])
df1 = df1[['datetime', 'price']]

df2 = df2.rename(columns={'entry_price': 'price', 'Position': 'position', 'Bid': 'bid' , 'Ask': 'ask'})
df2 = df2[['datetime', 'price', 'bid', 'ask', 'position']]

# Convert 'datetime' columns to datetime objects with consistent timezone
df1['datetime'] = pd.to_datetime(df1['datetime'], utc=True)
df2['datetime'] = pd.to_datetime(df2['datetime'], utc=True)

# Round datetime to the 3rd milisecond to ensure consistency
df1['datetime'] = df1['datetime'].dt.round('3ms')
df2['datetime'] = df2['datetime'].dt.round('3ms')

# Check for and remove duplicate datetime entries
df1 = df1.drop_duplicates(subset='datetime')
df2 = df2.drop_duplicates(subset='datetime')

# Perform a full outer join to retain all rows from both DataFrames
merged_df = pd.merge(df1, df2, on='datetime', how='outer', suffixes=('_sc', '_python'))

# Sort merged DataFrame by datetime
merged_df = merged_df.sort_values('datetime').reset_index(drop=True)

# Drop the 'source' columns
merged_df = merged_df.drop(columns=['source_sc', 'source_python'], errors='ignore')

# Combine rows with the same datetime value
combined_df = merged_df.groupby('datetime').agg({
    'price_sc': 'mean',
    'price_python': 'mean',
    'bid': 'mean',
    'ask': 'mean',
    'position': 'first'
}).reset_index()

# Create the 'match' column based on both price_sc and price_python being non-null
combined_df['match'] = combined_df['price_sc'].notna() & combined_df['price_python'].notna()

# Create the 'price_difference_check' column based on price_sc and price_python being different
combined_df['price_difference_check'] = (combined_df['price_sc'].notna() & combined_df['price_python'].notna() & (combined_df['price_sc'] != combined_df['price_python']))

# Create the 'price_python_adj' column based on conditions
combined_df['price_python_adj'] = combined_df.apply(
    lambda row: row['price_python'] - 0.25 if row['price_difference_check'] and row['position'] == -1
    else (row['price_python'] + 0.25 if row['price_difference_check'] and row['position'] == 1 else row['price_python']),
    axis=1
)

# Add a column that checks if 'price_python_adj' is equal to 'price_sc'
combined_df['price_python_adj_match'] = combined_df['price_python_adj'] == combined_df['price_sc']

# Save the combined DataFrame to a Parquet file
combined_df.to_parquet(output_parquet_path, index=False)
print(f"Filtered comparison data saved to {output_parquet_path}")



# Generate Trade Statistics
stats = {}

# Similar trade tally (both datetime and price match)
similar_trades = combined_df.dropna(subset=['price_sc', 'price_python'])
# if match is true then add to tally
similar_trades = similar_trades['match'] == True
stats['Similar Trades'] = len(similar_trades)

# Number of trades that do not match
no_match_trades = combined_df[~combined_df['match']]
stats['Un Matched Trades'] = len(no_match_trades)

# Total number of trades in combined DataFrame
total_trades = len(combined_df)
stats['Total Number of Trades'] = total_trades

# Different tally (prices are different even though datetime matches)
different_price_trades = combined_df[(combined_df['price_sc'].notna()) & 
                                     (combined_df['price_python'].notna()) & 
                                     (combined_df['price_sc'] != combined_df['price_python'])]
stats['Different Price Tally'] = len(different_price_trades)

# Percentage of trades that are the same
stats['Percentage the Same (time)'] = (stats['Similar Trades'] / total_trades) * 100 if total_trades > 0 else 0

stats['Percentage Different (price)'] = (stats['Different Price Tally'] / total_trades) * 100 if total_trades > 0 else 0

# Total profit/loss calculations (assuming price differences indicate profit/loss)
combined_df['price_difference'] = combined_df['price_sc'].fillna(0) - combined_df['price_python'].fillna(0)

#price difference after 

# Convert stats to DataFrame for saving
stats_df = pd.DataFrame(list(stats.items()), columns=['Metric', 'Value'])

# Save stats to Parquet file
stats_df.to_parquet(stats_parquet_path, index=False)

print(f"Trade statistics saved to {stats_parquet_path}")
