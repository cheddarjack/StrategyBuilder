   
import pandas as pd
import numpy as np
import math

# Provided complex value calculation function
def process_tick_data(tick_data):
    num_changes_5 = 5
    num_changes_15 = 15

    # Compute elapsed times between price changes for avg_rate_5 and avg_rate_15
    tick_data['price_changed'] = tick_data['Last'].diff() != 0
    change_times = tick_data.loc[tick_data['price_changed'], 'datetime']
    elapsed_times = change_times.diff().dt.total_seconds() * 1000  # in milliseconds

    tick_data.loc[tick_data['price_changed'], 'elapsed'] = elapsed_times

    # Create a DataFrame that only holds lines with a 'price_changed' value of True
    price_changed_df = tick_data[tick_data['price_changed']].copy()

    # Rolling averages of elapsed times within the price_changed_df
    price_changed_df['AvgRate5'] = price_changed_df['elapsed'].rolling(window=num_changes_5, min_periods=1).mean()
    price_changed_df['AvgRate15'] = price_changed_df['elapsed'].rolling(window=num_changes_15, min_periods=1).mean()

    # Forward-fill the AvgRate5 and AvgRate15 within the price_changed_df
    price_changed_df['AvgRate5'] = price_changed_df['AvgRate5'].ffill()
    price_changed_df['AvgRate15'] = price_changed_df['AvgRate15'].ffill()

    # Apply the computed AvgRate5 and AvgRate15 back to the original tick_data DataFrame
    tick_data.loc[price_changed_df.index, 'AvgRate5'] = price_changed_df['AvgRate5']
    tick_data.loc[price_changed_df.index, 'AvgRate15'] = price_changed_df['AvgRate15']

    # Compute predicted_15s
    tick_data['Predicted15s'] = np.where(
        tick_data['AvgRate15'] != 0,
        15000 / tick_data['AvgRate15'],
        0
    )

    return tick_data
