import pandas as pd
import pyarrow.parquet as pq
import os
import numpy as np

# Read the large Parquet file
table = pq.read_table('/home/cheddarjackk/StrategyBuilder/data/3_preprocessed_data/preprocessed_MES_March-Nov21.parquet')

# Convert to pandas DataFrame
df = table.to_pandas()

# Ensure 'datetime' column is in datetime format
df['datetime'] = pd.to_datetime(df['datetime'])

# Output directory
output_dir = '/home/cheddarjackk/StrategyBuilder/data/3_weeks_preprocessed'
os.makedirs(output_dir, exist_ok=True)

# Set the base date as a Friday at 21:00:00 before your data starts
# Adjust '2021-01-01' to a date prior to the earliest date in your dataset
base_date = pd.Timestamp('2021-01-01 21:00:00')

# Compute the time difference from base_date in seconds
tdelta = df['datetime'] - base_date

# Compute the number of weeks since base_date
week_seconds = 7 * 24 * 3600  # number of seconds in a week
df['week_number'] = (tdelta.dt.total_seconds() // week_seconds).astype(int)

# Compute the week ending date for each row
df['week_end'] = base_date + pd.to_timedelta(df['week_number'] + 1, unit='W')

# Group by 'week_number' and save each group
for week_num, group in df.groupby('week_number'):
    week_end = group['week_end'].iloc[0]
    week_str = week_end.strftime('%Y-%m-%d_%H%M%S')
    output_file = os.path.join(output_dir, f'data_week_end_{week_str}.parquet')
    # Drop the helper columns before saving
    group = group.drop(columns=['week_number', 'week_end'])
    # Save the group to Parquet file
    group.to_parquet(output_file, index=False)
    print(f"Saved week ending on {week_str} to {output_file}")