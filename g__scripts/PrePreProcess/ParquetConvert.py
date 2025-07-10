import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load text data into a pandas DataFrame
# Adjust the delimiter parameter as needed (e.g., ',' for comma, '\t' for tab)
df = pd.read_csv('C:\\Users\\jmorr\\OneDrive\\Desktop\\SierraChart\\Data\\MESu24_BarData.txt', delimiter=',')

# Print the loaded DataFrame to verify
print("Loaded DataFrame:")
print(df.head())

# Convert the pandas DataFrame to an Apache Arrow Table
table = pa.Table.from_pandas(df)

# Write the Apache Arrow Table to a Parquet file
pq.write_table(table, 'C:\\Projects\\StrategyBuilder\\data\\MESu24_BarData.parquet')

print("Text data successfully converted to Parquet file.")

#python Parquetconvert.py