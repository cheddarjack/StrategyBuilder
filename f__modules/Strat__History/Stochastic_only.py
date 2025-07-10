# ####################
# ####################
# This is an in depth look at the Stochastic Oscillator indicator.
# add to your code for debugging for needs of getting back to working order
# ###################
# ###################

import pandas as pd
import numpy as np
import math
import time

# Provided complex value calculation function
# def calculate_complex_value(average_size, x1, y1, x2, y2, threshold, min_value):
#     if x1 == x2:
#         return y1
#     m = (y2 - y1) / (x2 - x1)
#     b = y1 - m * x1
#     complex_value = m * average_size + b
#     if 8 < average_size < threshold:
#         complex_value = min_value + (complex_value - min_value) * math.exp(-(average_size - 8))
#     elif average_size >= threshold:
#         complex_value = min_value
#     if complex_value < min_value:
#         complex_value = min_value
#     return complex_value

# def process_tick_data(tick_data):
#     bar_size_ms = 15000  # 15 seconds in milliseconds
#     k_length = 10  # Fast %K length (last 9 completed bars)
#     D_Length = 3  # Slow %K length (last 3 Fast %K values)
#     candles = 6

#     # Pre-set for bars DataFrame
#     tick_data['bar_start'] = tick_data['Volume'].isna()
#     tick_data['bar_id'] = tick_data['bar_start'].cumsum()
#     tick_data['bar_count'] = tick_data.groupby('bar_id').cumcount() + 1

#     # Initialize columns for cumulative high and low values
#     tick_data['cumulative_high'] = tick_data.groupby('bar_id')['Last'].expanding().max().reset_index(level=0, drop=True)
#     tick_data['cumulative_low'] = tick_data.groupby('bar_id')['Last'].expanding().min().reset_index(level=0, drop=True)

#     # Aggregate tick data into OHLC bars
#     ohlc = tick_data.groupby('bar_id').agg(
#         Open=('Last', 'first'),
#         High=('Last', 'max'),
#         Low=('Last', 'min'),
#         Close=('Last', 'last')
#     )
#     bar_times = tick_data.groupby('bar_id')['datetime'].first()

#     # Create the bars DataFrame
#     bars = ohlc.reset_index().merge(bar_times.reset_index(), on='bar_id')

#     # Compute highest high and lowest low for each bar using the last 9 bars
#     bars['highest_high'] = bars['High'].rolling(window=k_length, min_periods=1).max()
#     bars['lowest_low'] = bars['Low'].rolling(window=k_length, min_periods=1).min()

#     # Assign 'highest_high' and 'lowest_low' to each tick using indexed lookups
#     tick_data = tick_data.merge(bars[['bar_id', 'highest_high', 'lowest_low']], on='bar_id', how='left')

#     # Calculate the current price percentage for each tick (Fast %K)
#     price_range = tick_data['highest_high'] - tick_data['lowest_low']
#     tick_data['Fast_K'] = np.where(
#         price_range != 0,
#         (tick_data['Last'] - tick_data['lowest_low']) / price_range * 100,
#         50  # Default value if no price movement
#     )

#     # Calculate the average price percentage for each tick using...
#     # Fast_K of - (Bar_ID -1, Bar_Count max), (Bar_ID -2, Bar_Count max), (Current tick_data value)
    
#     # Extract the Final_Fast_K for the last tick of each bar (where bar_count is max)
#     last_tick_fast_k = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K']]
#     last_tick_fast_k = last_tick_fast_k.rename(columns={'Fast_K': 'Final_Fast_K'})

#     # Add the Final_Fast_K back to tick_data for reference
#     tick_data = tick_data.merge(last_tick_fast_k, on='bar_id', how='left')

#     # Shift Final_Fast_K to get the values for the previous two bars
#     bars = bars.merge(last_tick_fast_k[['bar_id', 'Final_Fast_K']], on='bar_id', how='left')
#     bars['Fast_K_previous_1'] = bars['Final_Fast_K'].shift(1)
#     bars['Fast_K_previous_2'] = bars['Final_Fast_K'].shift(2)

#     # Merge the previous Fast_K values back into tick_data
#     tick_data = tick_data.merge(bars[['bar_id', 'Fast_K_previous_1', 'Fast_K_previous_2']], on='bar_id', how='left')

#     # Calculate Slow_K as the average of Fast_K for the current tick and previous two bars
#     tick_data['Slow_K'] = tick_data[['Fast_K', 'Fast_K_previous_1', 'Fast_K_previous_2']].mean(axis=1)

#     # Drop temporary columns
#     tick_data = tick_data.drop(['bar_start', 'Final_Fast_K'], axis=1)

#     return tick_data


def process_tick_data(tick_data,params):
    k_length = params['k_length']

    # Claculation of Stochastic Oscillator, refer to the following path for comments:
    # C:\Projects\StrategyBuilder\modules\Strat__History\Stochastic_only.py
    Stoch_time = time.time()

    tick_data['bar_start'] = tick_data['Volume'].isna()
    tick_data['bar_id'] = tick_data['bar_start'].cumsum()
    tick_data['bar_count'] = tick_data.groupby('bar_id').cumcount() + 1

    tick_data['cumulative_high'] = tick_data.groupby('bar_id')['Last'].cummax() #.expanding().max().reset_index(level=0, drop=True) is less efficient
    tick_data['cumulative_low'] = tick_data.groupby('bar_id')['Last'].cummin() # same inefficiency as above but with .min()

    # 3. Compute high_low_diff
    tick_data['high_low_diff'] = tick_data['cumulative_high'] - tick_data['cumulative_low']

    # 4. Aggregate OHLC and datetime in a single groupby
    ohlc_bars = tick_data.groupby('bar_id').agg(
        Open=('Last', 'first'),
        High=('Last', 'max'),
        Low=('Last', 'min'),
        Close=('Last', 'last'),
        datetime=('datetime', 'first')
    ).reset_index()

    # 5. Calculate rolling highest_high and lowest_low
    ohlc_bars['highest_high'] = ohlc_bars['High'].rolling(window=k_length, min_periods=1).max()
    ohlc_bars['lowest_low'] = ohlc_bars['Low'].rolling(window=k_length, min_periods=1).min()

    # 6. Map highest_high and lowest_low back to tick_data
    ohlc_bars.set_index('bar_id', inplace=True)
    tick_data['highest_high'] = tick_data['bar_id'].map(ohlc_bars['highest_high'])
    tick_data['lowest_low'] = tick_data['bar_id'].map(ohlc_bars['lowest_low'])

    # 7. Calculate Fast_K
    price_range = tick_data['highest_high'] - tick_data['lowest_low']
    tick_data['Fast_K'] = np.where(
        price_range != 0,
        (tick_data['Last'] - tick_data['lowest_low']) / price_range * 100,
        50  # Default value if no price movement
    )

    # 8. Get Final_Fast_K for each bar
    last_tick_fast_k = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K']]
    last_tick_fast_k = last_tick_fast_k.rename(columns={'Fast_K': 'Final_Fast_K'})
    last_tick_fast_k.set_index('bar_id', inplace=True)

    # 9. Merge Final_Fast_K back to ohlc_bars
    ohlc_bars = ohlc_bars.join(last_tick_fast_k, how='left')

    # 10. Calculate previous Fast_K values
    ohlc_bars['Fast_K_previous_1'] = ohlc_bars['Final_Fast_K'].shift(1)
    ohlc_bars['Fast_K_previous_2'] = ohlc_bars['Final_Fast_K'].shift(2)

    # 11. Map previous Fast_K values back to tick_data
    tick_data['Fast_K_previous_1'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_1'])
    tick_data['Fast_K_previous_2'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_2'])

    # 12. Calculate Slow_K
    tick_data['Slow_K'] = tick_data[['Fast_K', 'Fast_K_previous_1', 'Fast_K_previous_2']].mean(axis=1)

    # 13. Drop unnecessary columns early to save memory
    tick_data.drop([
        'Volume', 'bar_start', 'cumulative_high', 'cumulative_low',
        'highest_high', 'lowest_low', 'Fast_K_previous_1',
        'Fast_K_previous_2', 'Final_Fast_K', 'Fast_K'
    ], axis=1, inplace=True, errors='ignore')
    


    Stoch_time_finished = time.time() - Stoch_time
    print(f'Stochastic Oscillator calculation finished. Elapsed time: {Stoch_time_finished:.2f} seconds')

