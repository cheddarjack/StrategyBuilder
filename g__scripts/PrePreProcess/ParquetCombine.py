import pandas as pd

# Load the parquet files into DataFrames
df1 = pd.read_parquet('/home/cheddarjackk/Developer/StrategyBuilder/d__data/2_snipped_data/MESm24-Snipped.parquet')
df2 = pd.read_parquet('/home/cheddarjackk/Developer/StrategyBuilder/d__data/2_snipped_data/MESu24-Snipped.parquet')
df3 = pd.read_parquet('/home/cheddarjackk/Developer/StrategyBuilder/d__data/2_snipped_data/MESz24-Snipped.parquet')
df4 = pd.read_parquet('/home/cheddarjackk/Developer/StrategyBuilder/d__data/2_snipped_data/MESh25-Snipped.parquet')

# Print the columns of each DataFrame to verify they are the same
print("DataFrame 1 columns:")
print(df1.columns)
print("DataFrame 2 columns:")
print(df2.columns)
print("DataFrame 3 columns:")
print(df3.columns)
print("DataFrame 4 columns:")
print(df4.columns)

# Concatenate the DataFrames
combined_df = pd.concat([df1, df2, df3, df4])

# Save the combined DataFrame to a new Parquet file
combined_df.to_parquet('/home/cheddarjackk/Developer/StrategyBuilder/d__data/2_snipped_data/MES-March2Present.parquet')

print("Combined data saved to Parquet file.")