import numpy as np
import pandas as pd
import time

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
    print(f'Stochastic Oscillator calculation finished. Elapsed time: {Stoch_time_finished:.2f} seconds')

    # Compute Average rate of the last 5 and 15 price changes
    # C:\Projects\StrategyBuilder\modules\Strat__History\AvgRate_only.py
    changes_time = time.time()
    
       # Compute Average Rate including periods of unchanged price
    num_changes_5 = params['num_changes_5']

    tick_data['price_changed'] = tick_data['Last'].diff() != 0

    # Calculate cumulative price change count
    tick_data['price_change_count'] = tick_data['price_changed'].cumsum()

    # Map price change counts to datetimes
    price_change_times = tick_data.loc[tick_data['price_changed'], ['price_change_count', 'datetime']]
    price_change_times = price_change_times.set_index('price_change_count')['datetime']

    # For AvgRate5
    N = num_changes_5
    tick_data['price_change_count_N'] = tick_data['price_change_count'] - N
    tick_data['datetime_N'] = tick_data['price_change_count_N'].map(price_change_times)

    tick_data['time_since_Nth_price_change'] = (
        tick_data['datetime'] - tick_data['datetime_N']
    ).dt.total_seconds() * 1000  # in milliseconds

    # Handle NaT values
    tick_data['time_since_Nth_price_change'] = tick_data['time_since_Nth_price_change'].ffill()

    tick_data['AvgRate5'] = tick_data['time_since_Nth_price_change'] / N
    tick_data['AvgRate5'] = tick_data['AvgRate5'].ffill()

    tick_data['AvgRate5_at_price_change'] = np.nan
    price_change_indices = tick_data.index[tick_data['price_changed']]

    # Assign 'AvgRate5' values at price change indices
    tick_data.loc[price_change_indices, 'AvgRate5_at_price_change'] = tick_data.loc[price_change_indices, 'AvgRate5']

    # Shift the 'AvgRate5_at_price_change' column
    tick_data['AvgRate5_at_previous_price_change'] = tick_data['AvgRate5_at_price_change'].shift(1)

    # Forward-fill to propagate previous price change values
    tick_data['AvgRate5_at_previous_price_change'] = tick_data['AvgRate5_at_previous_price_change'].ffill()

    # Calculate AvgRate5_RoC from the previous price change
    tick_data['AvgRate5_RoC'] = (
        (tick_data['AvgRate5'] - tick_data['AvgRate5_at_previous_price_change']) /
        tick_data['AvgRate5_at_previous_price_change'].replace(0, np.nan)
    ) * 100

    # Handle division by zero or NaN values
    tick_data['AvgRate5_RoC'] = tick_data['AvgRate5_RoC'].fillna(0)

    # **Calculate 'is_slowing_down'**
    slowing_threshold = params.get('AvgRate5_RoC_Slowing_Threshold', 50)  # Adjust threshold as needed
    tick_data['is_slowing_down'] = tick_data['AvgRate5_RoC'] > slowing_threshold

    # Optional: Remove temporary columns
    tick_data.drop([
        'AvgRate5_at_price_change', 'AvgRate5_at_previous_price_change', 'price_changed', 
        'price_change_count', 'price_change_count_N', 'datetime_N', 'time_since_Nth_price_change'
    ], axis=1, inplace=True)

    
    changes_time_finished = time.time() - changes_time
    print(f'Average rate calculation finished. Elapsed time: {changes_time_finished:.2f} seconds')

    # Compute bar sizes and average size over last 6 candles
    # C:\Projects\StrategyBuilder\modules\Strat__History\ComplexValue.py
    Complex_time = time.time()
    candles = params['candles']
    high_average = params['high_average']
    high_value = params['high_value']
    low_average = params['low_average']
    low_value = params['low_value']
    threshold = params['threshold']
    min_value = params['min_value']
    max_value = params['max_value']

    max_bar_count_indices = tick_data.groupby('bar_id')['bar_count'].idxmax()
    max_bar_diff = tick_data.loc[max_bar_count_indices, ['bar_id', 'high_low_diff']].reset_index(drop=True)
    max_bar_diff['average_size'] = max_bar_diff['high_low_diff'].rolling(window=candles, min_periods=candles).mean().shift(1) * 4

    tick_data = tick_data.merge(max_bar_diff[['bar_id', 'average_size']], on='bar_id', how='left')
    tick_data['average_size'] = tick_data.groupby('bar_id')['average_size'].ffill()

    tick_data['ComplexValue'] = tick_data['average_size'].apply(
        lambda avg_size: calculate_complex_value(avg_size, high_average, high_value, low_average, low_value, threshold, min_value, max_value)
    )

    tick_data.drop([
        'bar_start', 'cumulative_high', 'cumulative_low', 'high_low_diff',
        'highest_high', 'lowest_low', 'Fast_K_previous_1', 'average_size', 'Fast_K_previous_2',
        'Fast_K_previous_2', 'Final_Fast_K', 'Fast_K', 'prev_highest_high', 'prev_lowest_low', 
        ], axis=1, inplace=True, errors='ignore')

    Complex_time_finished = time.time() - Complex_time
    print(f'Complex value calculation finished. Elapsed time: {Complex_time_finished:.2f} seconds')


    return tick_data


def generate_signals(df, params):

    def time_to_seconds(time_str):
        t = pd.to_datetime(time_str).time()
        return t.hour * 3600 + t.minute * 60 + t.second

    start_times = [time_to_seconds(params[f'start_time{i}']) for i in range(1, 4)]
    end_times = [time_to_seconds(params[f'end_time{i}']) for i in range(1, 4)]

    time_in_sec = ((df['datetime'].values.astype('datetime64[s]') - df['datetime'].values.astype('datetime64[D]')).astype('int'))
    within_timeframe = np.zeros(len(df), dtype=bool)
    for start, end in zip(start_times, end_times):
        if start <= end:
            within_timeframe |= (time_in_sec >= start) & (time_in_sec <= end)
        else:
            within_timeframe |= (time_in_sec >= start) | (time_in_sec <= end)

    df['state'] = 0  # Start in State 0

    # Create a flag for when AvgRate5 < ComplexValue
    df['fast_moving'] = df['AvgRate5'] < df['ComplexValue']

    # Initialize the 'in_fast_moving' state
    df['state'] = 0  # State 0: Neutral
    df['state'] = np.where(df['fast_moving'], 1, df['state'])  # State 1: Fast-moving

    # Forward-fill the state to maintain State 1 until slowing down occurs
    df['state'] = df['state'].replace(0, np.nan).ffill().fillna(0)

    # Identify slowing down in State 1
    df['slowing_down'] = (df['state'] == 1) & df['is_slowing_down']

    # Transition to State 2 when slowing down is detected
    df.loc[df['slowing_down'], 'state'] = 2

    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Your existing trigger conditions
    df['buy_trigger'] = (
        (df['Slow_K'] < params['Slow_K_Low']) &
        (df['state'] == 2) &
        (df['ComplexValue'] != params['min_value']) &
        (df['bar_id'] > 9) &
        (df['Volume'].notna())
    )

    df['sell_trigger'] = (
        (df['Slow_K'] > params['Slow_K_High']) &
        (df['state'] == 2) &
        (df['ComplexValue'] != params['min_value']) &
        (df['bar_id'] > 9) &
        (df['Volume'].notna())
    )

    # Combine buy and sell triggers into a single 'trade_signal' column
    df['trade_signal'] = df['buy_trigger'] | df['sell_trigger']

    # Initialize variables
    signal_indices = df.index[df['trade_signal']].tolist()
    accepted_indices = []
    last_accepted_time = None

    # Iterate over the indices where 'trade_signal' is True
    for idx in signal_indices:
        current_time = df.loc[idx, 'datetime']
        if last_accepted_time is None or (current_time - last_accepted_time).total_seconds() >= 120:
            accepted_indices.append(idx)
            last_accepted_time = current_time
        else:
            # Skip this signal, too soon after the last accepted signal
            pass

    # Create an 'allowed' column to mark accepted signals
    df['allowed'] = False
    df.loc[accepted_indices, 'allowed'] = True

    # Update the trigger conditions to include the new time constraint
    df['buy_signal'] = df['buy_trigger'] & df['allowed']
    df['sell_signal'] = df['sell_trigger'] & df['allowed']

    # Remove intermediate columns
    df.drop(['buy_trigger', 'sell_trigger', 'trade_signal', 'allowed'], axis=1, inplace=True)

    return df

def simulate_trades(df,params):
    df['position'] = 0
    df['entry_price'] = np.nan
    df['exit_price'] = np.nan
    df['ticks'] = 0.0
    df['cumulative_PL'] = 0.0
    tick_size = 0.25
    tick_price = 1.25

    take_profit = params['take_profit']
    stop_loss = params['stop_loss']

    df['position'] = np.where(df['buy_signal'], 1,
                              np.where(df['sell_signal'], -1, np.nan))
    df['position'] = df['position'].ffill().fillna(0)

    entry_points = (df['buy_signal'] | df['sell_signal']) & (df['position'] != 0)

    df.loc[entry_points, 'entry_price'] = df['Last']
    df['entry_price'] = df['entry_price'].where(df['position'] != 0).ffill()

    df['SL_price_long'] = df['entry_price'] - stop_loss
    df['TP_price_long'] = df['entry_price'] + take_profit
    df['SL_price_short'] = df['entry_price'] + stop_loss
    df['TP_price_short'] = df['entry_price'] - take_profit

    df['trade_id'] = df['buy_signal'].cumsum() + df['sell_signal'].cumsum()
    df['trade_id'] = df['trade_id'].where(df['position'] != 0)

    exit_condition_long = (df['position'] == 1) & (
        (df['Last'] <= df['SL_price_long']) | (df['Last'] >= df['TP_price_long']) & (df['Volume'].notna())
    )
    exit_condition_short = (df['position'] == -1) & (
        (df['Last'] >= df['SL_price_short']) | (df['Last'] <= df['TP_price_short']) & (df['Volume'].notna())
    )
    df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)
    df['time_elapsed'] = df.groupby('trade_id')['time_diff'].cumsum()

    exit_due_to_time = (df['time_elapsed'] >= params['stop_timer']) & (df['position'] != 0) & (df['Volume'].notna())

    exit_condition = (exit_condition_long | exit_condition_short | exit_due_to_time).fillna(False).astype(bool)
    df['exit_condition'] = exit_condition  # Store for reference

    cummax_exit_condition = df.groupby('trade_id')['exit_condition'].cummax().fillna(False).astype(bool)

    shifted_cummax = cummax_exit_condition.shift(fill_value=False)
    not_shifted_cummax = ~shifted_cummax

    df['exit_flag'] = exit_condition & not_shifted_cummax
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])

    df.loc[df['exit_flag'], 'exit_price'] = df.loc[df['exit_flag'], 'Last']

    df.loc[df['exit_flag'], 'ticks'] = np.where(
        df.loc[df['exit_flag'], 'position'] == 1,
        df.loc[df['exit_flag'], 'exit_price'] - df.loc[df['exit_flag'], 'entry_price'],
        df.loc[df['exit_flag'], 'entry_price'] - df.loc[df['exit_flag'], 'exit_price']
    )

    df['ticks'] = df['ticks'].fillna(0)
    df['profit_loss'] = np.where(df['exit_flag'] == True,((df['ticks'] / tick_size) * tick_price) -  1.04, 0)
    df['cumulative_PL'] = df['profit_loss'].cumsum()

    # drop intermediate columns
    df.drop([ 'time_diff', 'exit_condition'], axis=1, inplace=True)

    return df

def extract_trades(tick_data):

    trade_entries_exits = tick_data[(tick_data['buy_signal']) | (tick_data['sell_signal']) | (tick_data['exit_flag'])]
    trade_entries_exits = trade_entries_exits.sort_values(by='datetime')

    return trade_entries_exits

def calculate_trade_metrics(trades_data):
    # Precompute necessary boolean masks
    profit_positive = trades_data['profit_loss'] > 0
    profit_negative = trades_data['profit_loss'] < 0
    profit_zero = trades_data['profit_loss'] == 0
    exit_flag = trades_data['exit_flag'] == True
    position_long = trades_data['position'] == 1
    position_short = trades_data['position'] == -1
    time_elapsed_ge_120 = trades_data['time_elapsed'] >= 120

    # Calculate unique trade IDs once
    unique_trade_ids = trades_data['trade_id'].unique()

    # Total trades
    total_trades = len(unique_trade_ids) -1

    # Winning trades
    winning_trade_ids = trades_data.loc[profit_positive, 'trade_id'].unique()
    winning_trades = len(winning_trade_ids)

    # Losing trades
    losing_trade_ids = trades_data.loc[profit_negative, 'trade_id'].unique()
    losing_trades = len(losing_trade_ids)

    # Total zero-point trades
    zero_point_trade_ids = trades_data.loc[exit_flag & profit_zero, 'trade_id'].unique()
    total_zero_point_trades = len(zero_point_trade_ids)

    # Long trades
    long_trade_ids = trades_data.loc[position_long, 'trade_id'].unique()
    long_trades = len(long_trade_ids)

    # Short trades
    short_trade_ids = trades_data.loc[position_short, 'trade_id'].unique()
    short_trades = len(short_trade_ids)

    # Winning long trades
    winning_long_trade_ids = trades_data.loc[position_long & profit_positive, 'trade_id'].unique()
    winning_long_trades = len(winning_long_trade_ids)

    # Losing long trades
    losing_long_trade_ids = trades_data.loc[position_long & profit_negative, 'trade_id'].unique()
    losing_long_trades = len(losing_long_trade_ids)

    # Winning short trades
    winning_short_trade_ids = trades_data.loc[position_short & profit_positive, 'trade_id'].unique()
    winning_short_trades = len(winning_short_trade_ids)

    # Losing short trades
    losing_short_trade_ids = trades_data.loc[position_short & profit_negative, 'trade_id'].unique()
    losing_short_trades = len(losing_short_trade_ids)

    # Total profit/loss
    total_profit_loss = trades_data['profit_loss'].sum()

    # Total commissions
    total_commissions = total_trades * 1.04  # Assuming $1.04 commission per trade

    # Total time cut trades
    time_cut_trade_ids = trades_data.loc[exit_flag & time_elapsed_ge_120, 'trade_id'].unique()
    total_time_cut_trades = len(time_cut_trade_ids)

    # Win percentage
    win_percent = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

    # Average profit per trade
    avg_profit_per_trade = total_profit_loss / total_trades if total_trades > 0 else 0

    # Average profit per winning trade
    if winning_trades > 0:
        avg_profit_per_winning_trade = trades_data.loc[profit_positive, 'profit_loss'].sum() / winning_trades
    else:
        avg_profit_per_winning_trade = 0

    # Average loss per losing trade
    if losing_trades > 0:
        avg_loss_per_losing_trade = trades_data.loc[profit_negative, 'profit_loss'].sum() / losing_trades
    else:
        avg_loss_per_losing_trade = 0
    
    largest_winner = trades_data.loc[profit_positive, 'profit_loss'].max()
    largest_loser = trades_data.loc[profit_negative, 'profit_loss'].min()
    wins_over_5 = trades_data.loc[profit_positive, 'profit_loss'][trades_data.loc[profit_positive, 'profit_loss'] > 5].count()
    losses_over_10 = trades_data.loc[profit_negative, 'profit_loss'][trades_data.loc[profit_negative, 'profit_loss'] < -10].count()
    longest_trade_when_exit = trades_data.loc[exit_flag, 'time_elapsed'].max()
    shortest_trade_when_exit = trades_data.loc[exit_flag, 'time_elapsed'].min()
    most_ticks = trades_data['ticks'].max()

    # Gross profit
    gross_profit = trades_data.loc[profit_positive, 'profit_loss'].sum()
    gross_loss = abs(trades_data.loc[profit_negative, 'profit_loss'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Compile metrics into a dictionary
    metrics = {
        'Total Trades': total_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': (losing_trades + total_zero_point_trades),
        'Long Trades': long_trades,
        'Short Trades': short_trades,
        'Winning Long Trades': winning_long_trades,
        'Losing Long Trades': losing_long_trades,
        'Winning Short Trades': winning_short_trades,
        'Losing Short Trades': losing_short_trades,
        'Total Profit/Loss': total_profit_loss,
        'Total Commissions': total_commissions,
        'Total Time Cut Trades': total_time_cut_trades,
        'Win Percentage': win_percent,
        'Average Profit per Trade': avg_profit_per_trade,
        'Average Profit per Winning Trade': avg_profit_per_winning_trade,
        'Average Loss per Losing Trade': avg_loss_per_losing_trade,
        'Largest Winner': largest_winner,
        'Largest Loser': largest_loser,
        'Wins over 5': wins_over_5,
        'Losses under -10': losses_over_10,
        'Longest Trade when Exit': longest_trade_when_exit,
        'Shortest Trade when Exit': shortest_trade_when_exit,
        'Most Ticks': most_ticks,
        'Gross Profit': gross_profit,
        'Profit Factor': profit_factor
    }

    return metrics


