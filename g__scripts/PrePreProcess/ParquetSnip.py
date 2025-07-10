import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import yaml


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def snip_last_lines(df, x):
    # Snip the last x lines
    print(f"Snipping the last {x} lines...")
    df_snipped = df.iloc[:x]
    return df_snipped


def snip_first_lines(df, y):
    # Snip the first y lines
    print(f"Snipping the first {y} lines...")
    df_snipped = df.iloc[y:]
    return df_snipped


if __name__ == "__main__":
    # Load configuration
    config_path = '/home/cheddarjackk/Developer/StrategyBuilder/c__config/config.yaml'
    config = load_config(config_path)

    # Load parquet file from path specified in the config
    parquet_path = config['data']['converted_data_path']
    print(f"Loading parquet file from {parquet_path}...")
    table = pq.read_table(parquet_path)
    df = table.to_pandas()

    # Number of lines to snip from the end and the beginning
    x = 27328557  # Snip last lines
    y = 272117 # Snip first lines

    # Execute the snipping processes
    df = snip_last_lines(df, x)
    df = snip_first_lines(df, y)

    # Save the resulting DataFrame to another parquet file specified in the config
    output_path = config['data']['snipped_data_path']
    print(f"Saving the snipped data to {output_path}...")
    table_snipped = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table_snipped, output_path)

    print("Processing complete.")
