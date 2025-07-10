import numpy as np
import time
import pandas as pd

def process_tick_data(tick_data,params):
    k_length = params['k_length']

    # Claculation of Stochastic Oscillator, refer to the following path for comments:
    # C:\Projects\StrategyBuilder\modules\Strat__History\Stochastic_only.py
    Stoch_time = time.time()

    tick_data['bar_start'] = tick_data['Volume'].isna()
    tick_data['bar_id'] = tick_data['bar_start'].cumsum()
    tick_data['bar_count'] = tick_data.groupby('bar_id').cumcount() + 1

    # remove all lines where the volume is NaN
    tick_data.dropna(subset=['Volume'], inplace=True)

    tick_data['cumulative_high'] = tick_data.groupby('bar_id')['Last'].cummax() #.expanding().max().reset_index(level=0, drop=True) is less efficient
    tick_data['cumulative_low'] = tick_data.groupby('bar_id')['Last'].cummin() # same inefficiency as above but with .min()

    tick_data['high_low_diff'] = tick_data['cumulative_high'] - tick_data['cumulative_low']

    ohlc_bars = tick_data.groupby('bar_id').agg(
        Open=('Last', 'first'),
        High=('Last', 'max'),
        Low=('Last', 'min'),
        Close=('Last', 'last'),
        datetime=('datetime', 'first')
    ).reset_index()

    ohlc_bars['highest_high'] = ohlc_bars['High'].rolling(window=(k_length - 1), min_periods=1).max().shift()
    ohlc_bars['lowest_low'] = ohlc_bars['Low'].rolling(window=(k_length - 1), min_periods=1).min().shift()


    ohlc_bars.set_index('bar_id', inplace=True)
    tick_data['prev_highest_high'] = tick_data['bar_id'].map(ohlc_bars['highest_high'])
    tick_data['prev_lowest_low'] = tick_data['bar_id'].map(ohlc_bars['lowest_low'])

    tick_data['highest_high'] = np.maximum(tick_data['prev_highest_high'], tick_data['High'])
    tick_data['lowest_low'] = np.minimum(tick_data['prev_lowest_low'], tick_data['Low'])

    price_range = tick_data['highest_high'] - tick_data['lowest_low']
    tick_data['Fast_K'] = np.where(
        price_range != 0,
        (tick_data['Last'] - tick_data['lowest_low']) / price_range * 100,
        50  # Default value if no price movement
    )

    last_tick_fast_k = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K']]
    last_tick_fast_k = last_tick_fast_k.rename(columns={'Fast_K': 'Final_Fast_K'})
    last_tick_fast_k.set_index('bar_id', inplace=True)

    ohlc_bars = ohlc_bars.join(last_tick_fast_k, how='left')

    ohlc_bars['Fast_K_previous_1'] = ohlc_bars['Final_Fast_K'].shift(1)
    ohlc_bars['Fast_K_previous_2'] = ohlc_bars['Final_Fast_K'].shift(2)

    tick_data['Fast_K_previous_1'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_1'])
    tick_data['Fast_K_previous_2'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_2'])

    tick_data['Slow_K'] = tick_data[['Fast_K', 'Fast_K_previous_1', 'Fast_K_previous_2']].mean(axis=1) # Default smoothing factor of 3

    Stoch_time_finished = time.time() - Stoch_time
    # print(f'Stochastic Oscillator calculation finished. Elapsed time: {Stoch_time_finished:.2f} seconds')


    # Calculation of Stochastic Oscillator with k_length_2
    k_length_2 = params['k_length_2']  

    Stoch_time_2 = time.time()

    ohlc_bars['highest_high_2'] = ohlc_bars['High'].rolling(window=(k_length_2 - 1), min_periods=1).max().shift()
    ohlc_bars['lowest_low_2'] = ohlc_bars['Low'].rolling(window=(k_length_2 - 1), min_periods=1).min().shift()

    # Map 'highest_high_2' and 'lowest_low_2' back to tick_data
    tick_data['prev_highest_high_2'] = tick_data['bar_id'].map(ohlc_bars['highest_high_2'])
    tick_data['prev_lowest_low_2'] = tick_data['bar_id'].map(ohlc_bars['lowest_low_2'])

    tick_data['highest_high_2'] = np.maximum(tick_data['prev_highest_high_2'], tick_data['High'])
    tick_data['lowest_low_2'] = np.minimum(tick_data['prev_lowest_low_2'], tick_data['Low'])

    price_range_2 = tick_data['highest_high_2'] - tick_data['lowest_low_2']
    tick_data['Fast_K_2'] = np.where(
        price_range_2 != 0,
        (tick_data['Last'] - tick_data['lowest_low_2']) / price_range_2 * 100,
        50  # Default value if no price movement
    )

    last_tick_fast_k_2 = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K_2']]
    last_tick_fast_k_2 = last_tick_fast_k_2.rename(columns={'Fast_K_2': 'Final_Fast_K_2'})
    last_tick_fast_k_2.set_index('bar_id', inplace=True)

    # Join 'Final_Fast_K_2' back to 'ohlc_bars'
    ohlc_bars = ohlc_bars.join(last_tick_fast_k_2, how='left')

    ohlc_bars['Fast_K_previous_1_2'] = ohlc_bars['Final_Fast_K_2'].shift(1)
    ohlc_bars['Fast_K_previous_2_2'] = ohlc_bars['Final_Fast_K_2'].shift(2)

    tick_data['Fast_K_previous_1_2'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_1_2'])
    tick_data['Fast_K_previous_2_2'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_2_2'])

    tick_data['Slow_K_2'] = tick_data[['Fast_K_2', 'Fast_K_previous_1_2', 'Fast_K_previous_2_2']].mean(axis=1)  # Default smoothing factor of 3

    # drop columns
    tick_data.drop(['prev_highest_high_2', 'prev_lowest_low_2', 'highest_high_2', 'lowest_low_2', 'Fast_K_2', 
                    'Fast_K_previous_1_2', 'Fast_K_previous_2_2'], axis=1, inplace=True)

    Stoch_time_finished_2 = time.time() - Stoch_time_2
    # print(f'Stochastic Oscillator calculation with k_length_2 finished. Elapsed time: {Stoch_time_finished_2:.2f} seconds')



    # Calculation of Stochastic Oscillator with k_length_3
    k_length_3 = params['k_length_3']
    Stoch_time_3 = time.time()

    # Recalculate 'highest_high' and 'lowest_low' with k_length_3
    ohlc_bars['highest_high_3'] = ohlc_bars['High'].rolling(window=(k_length_3 - 1), min_periods=1).max().shift()
    ohlc_bars['lowest_low_3'] = ohlc_bars['Low'].rolling(window=(k_length_3 - 1), min_periods=1).min().shift()

    # Map 'highest_high_3' and 'lowest_low_3' back to tick_data
    tick_data['prev_highest_high_3'] = tick_data['bar_id'].map(ohlc_bars['highest_high_3'])
    tick_data['prev_lowest_low_3'] = tick_data['bar_id'].map(ohlc_bars['lowest_low_3'])

    tick_data['highest_high_3'] = np.maximum(tick_data['prev_highest_high_3'], tick_data['High'])
    tick_data['lowest_low_3'] = np.minimum(tick_data['prev_lowest_low_3'], tick_data['Low'])

    price_range_3 = tick_data['highest_high_3'] - tick_data['lowest_low_3']
    tick_data['Fast_K_3'] = np.where(
        price_range_3 != 0,
        (tick_data['Last'] - tick_data['lowest_low_3']) / price_range_3 * 100,
        50  # Default value if no price movement
    )

    last_tick_fast_k_3 = tick_data.loc[tick_data.groupby('bar_id')['bar_count'].idxmax(), ['bar_id', 'Fast_K_3']]
    last_tick_fast_k_3 = last_tick_fast_k_3.rename(columns={'Fast_K_3': 'Final_Fast_K_3'})
    last_tick_fast_k_3.set_index('bar_id', inplace=True)

    # Join 'Final_Fast_K_3' back to 'ohlc_bars'
    ohlc_bars = ohlc_bars.join(last_tick_fast_k_3, how='left')


    ohlc_bars['Fast_K_previous_1_3'] = ohlc_bars['Final_Fast_K_3'].rolling(window=59, min_periods=1).sum().shift(1)

    tick_data['Fast_K_previous_1_3'] = tick_data['bar_id'].map(ohlc_bars['Fast_K_previous_1_3'])

    tick_data['Slow_K_3'] = (tick_data[['Fast_K_3', 'Fast_K_previous_1_3']].sum(axis=1)) / 60 # Default smoothing factor of 3

    # drop columns
    tick_data.drop(['prev_highest_high_3', 'prev_lowest_low_3', 'highest_high_3', 'lowest_low_3', 'Fast_K_3',
                    'Fast_K_previous_1_3'], axis=1, inplace=True)
    

    Stoch_time_finished_3 = time.time() - Stoch_time_3
    # print(f'Stochastic Oscillator calculation with k_length_3 finished. Elapsed time: {Stoch_time_finished_3:.2f} seconds')



    return tick_data