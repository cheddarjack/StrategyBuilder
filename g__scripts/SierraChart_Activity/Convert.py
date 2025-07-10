import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Load text data into a pandas DataFrame
file_path = 'C:\\Users\\jmorr\\OneDrive\\Desktop\\SierraChart\\SavedTradeActivity\\TradeActivityLogExport_2024-11-12.txt'
# Define column names (from the sample provided)
column_names = [
    "ActivityType", "DateTime", "TransDateTime", "Symbol", "OrderActionSource", "InternalOrderID",
    "ServiceOrderID", "OrderType", "Quantity", "BuySell", "Price", "Price2", "OrderStatus",
    "FillPrice", "FilledQuantity", "TradeAccount", "OpenClose", "ParentInternalOrderID",
    "PositionQuantity", "FillExecutionServiceID", "HighDuringPosition", "LowDuringPosition", "Note",
    "AccountBalance", "ExchangeOrderID", "ClientOrderID", "TimeInForce", "Username"
]

# Load the txt file into a pandas DataFrame
df = pd.read_csv(file_path, delimiter='\t', names=column_names, skiprows=1)

def move_decimal(value):
    return value / 100 if pd.notna(value) else value

# Apply the function to the specified columns
for column in ['Price', 'FillPrice']:
    df[column] = df[column].apply(move_decimal)

# Initialize the accumulated profit/loss column and individual trade profit/loss column
df['AccumulatedPL'] = 0.0
df['TradePL'] = 0.0

# Iterate through the DataFrame to calculate profit/loss for each trade
accumulated_pl = 0.0
entry_price = None
entry_type = None

for i in range(len(df)):
    if pd.notna(df.at[i, 'PositionQuantity']):
        if df.at[i, 'PositionQuantity'] == 1:  # Long entry
            entry_price = df.at[i, 'FillPrice']
            entry_type = 'long'
        elif df.at[i, 'PositionQuantity'] == -1:  # Short entry
            entry_price = df.at[i, 'FillPrice']
            entry_type = 'short'
    else:  # Trade is closed
        if entry_price is not None:
            exit_price = df.at[i, 'FillPrice']
            if entry_type == 'long':
                trade_pl = exit_price - entry_price
            elif entry_type == 'short':
                trade_pl = entry_price - exit_price
            # Adjust for tick size and commission
            trade_pl = (trade_pl / 0.25) - 1.04
            accumulated_pl += trade_pl
            df.at[i, 'TradePL'] = trade_pl
            entry_price = None
            entry_type = None
    df.at[i, 'AccumulatedPL'] = accumulated_pl

# Convert the DataFrame to an Arrow Table
table = pa.Table.from_pandas(df)

# Drop unnecessary columns from the Arrow Table
table = table.drop(['OrderActionSource', 'InternalOrderID', 'ServiceOrderID', 'Note', 'TradeAccount',
                    'OrderStatus', 'OpenClose', 'ParentInternalOrderID', 'ActivityType',
                    'FillExecutionServiceID', 'HighDuringPosition', 'LowDuringPosition', 'AccountBalance', 'ExchangeOrderID',
                    'ClientOrderID', 'TimeInForce', 'Username', 'TransDateTime', 'FilledQuantity', 'Symbol', 'Price2', 'Quantity'])

# Write the table to a Parquet file
parquet_file_path = 'C:\\Projects\\StrategyBuilder\\data\\SC\\2024-11-12_L_August_SC.parquet'
pq.write_table(table, parquet_file_path)

print("Text data successfully converted to Parquet file.")