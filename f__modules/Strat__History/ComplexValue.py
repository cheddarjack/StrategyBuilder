import pandas as pd
import numpy as np
import math

def calculate_complex_value(average_size, low_average, low_value, high_average, high_value, threshold, min_value, max_value):
    if average_size <= 0:
        complex_value = max_value  # Starting value at average_size = 0
    elif 0 < average_size <= low_average:
        m_initial = (low_value - max_value) / (low_average - 0)
        b_initial = max_value  # At average_size = 0, complex_value = max_value
        complex_value = m_initial * average_size + b_initial
    elif low_average < average_size <= high_average:
        m_middle = (high_value - low_value) / (high_average - low_average)
        b_middle = low_value - m_middle * low_average
        complex_value = m_middle * average_size + b_middle
    elif high_average < average_size < threshold:
        m_decay = (min_value - high_value) / (threshold - high_average)
        b_decay = high_value - m_decay * high_average
        complex_value = m_decay * average_size + b_decay
    else:
        complex_value = min_value
    return complex_value


def process_tick_data(tick_data):
   
    candles = 6
    high_average = 9.0 # x1
    high_value = 200.0 # y1
    low_average = 2.0 # x2
    low_value = 1000.0 # y2
    threshold = 14.0
    min_value = 120.0
    max_value = 1500

    # Compute bar sizes and average size over last 6 candles

    # Extract the high_low_diff for the maximum bar_count of each bar_id
    max_bar_count_indices = tick_data.groupby('bar_id')['bar_count'].idxmax()
    max_bar_diff = tick_data.loc[max_bar_count_indices, ['bar_id', 'high_low_diff']].reset_index(drop=True)

    # Calculate the rolling average of the high_low_diff for the last 4 bars
    max_bar_diff['average_size'] = max_bar_diff['high_low_diff'].rolling(window=candles, min_periods=candles).mean().shift(1) * 4

    # Merge this average back into tick_data
    tick_data = tick_data.merge(max_bar_diff[['bar_id', 'average_size']], on='bar_id', how='left')
    tick_data['average_size'] = tick_data.groupby('bar_id')['average_size'].ffill()

    # Compute complex value for each bar
    tick_data['ComplexValue'] = tick_data['average_size'].apply(
        lambda avg_size: calculate_complex_value(avg_size, high_average, high_value, low_average, low_value, threshold, min_value, max_value)
    )

    return tick_data