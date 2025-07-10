import numpy as np
import pandas as pd
import time
from numba import njit

def generate_signals(df, params):

    start_times = [params[f'start_time{i}'] for i in range(1, 2)]
    end_times = [params[f'end_time{i}'] for i in range(1, 2)]

    # Calculate seconds since midnight for each datetime entry in the DataFrame
    df['time_in_sec'] = df['datetime'].dt.hour * 3600 + df['datetime'].dt.minute * 60 + df['datetime'].dt.second

    # Initialize a boolean mask for within timeframe
    within_timeframe = np.zeros(len(df), dtype=bool)
    for start, end in zip(start_times, end_times):
        if start <= end:
            within_timeframe |= (df['time_in_sec'] >= start) & (df['time_in_sec'] <= end)
        else:
            within_timeframe |= (df['time_in_sec'] >= start) | (df['time_in_sec'] <= end)


    # Assuming 'df' is your DataFrame
    # First, compute the 'fast_moving' flag
    df['fast_moving'] = df['AvgRate5'] < df['ComplexValue']

    # Convert necessary columns to NumPy arrays for Numba compatibility
    fast_moving = df['fast_moving'].to_numpy(dtype=np.bool_)
    is_slowing_down = df['is_slowing_down'].to_numpy(dtype=np.bool_)

    # Initialize an array to hold the state values
    state_array = np.zeros(len(df), dtype=np.int8)

    @njit
    def compute_state(fast_moving, is_slowing_down):
        state = np.zeros_like(fast_moving, dtype=np.int8)
        current_state = 0  # Start with state 0
        for i in range(len(fast_moving)):
            if current_state == 0:
                if fast_moving[i]:
                    current_state = 1
            elif current_state == 1:
                if is_slowing_down[i]:
                    current_state = 2
                elif not fast_moving[i]:
                    current_state = 0
            elif current_state == 2:
                if not is_slowing_down[i]:
                    if fast_moving[i]:
                        current_state = 1
                    else:
                        current_state = 0
                # Remain in state 2 if 'is_slowing_down' is still True
            state[i] = current_state
        return state
    
    # Call the Numba-optimized function
    state_array = compute_state(fast_moving, is_slowing_down)

    # Assign the computed state back to the DataFrame
    df['state'] = state_array

    # Ensure 'datetime' is in datetime format
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Your existing trigger conditions
    df['buy_trigger'] = (
        (df['Slow_K'] > params['Slow_K_Low']) &
        (df['Slow_K'] < params['Slow_K_Low_cap']) &
        (df['state'] == 2) &
        (df['ComplexValue'] != params['min_value']) &
        (df['ComplexValue'] > df['AvgRate5']) &
        (df['bar_id'] > 19) &
        (df['Volume'].notna()) &
        within_timeframe
    )

    df['sell_trigger'] = (
        (df['Slow_K'] > params['Slow_K_High']) &
        (df['Slow_K'] < params['Slow_K_High_cap']) &
        (df['state'] == 2) &
        (df['ComplexValue'] != params['min_value']) &
        (df['ComplexValue'] > df['AvgRate5']) &
        (df['bar_id'] > 19) &
        (df['Volume'].notna()) &
        within_timeframe
    )
    df.drop(['bar_id'], axis=1, inplace=True)

    # Combine buy and sell triggers into a single 'trade_signal' column
    df['trade_signal'] = df['buy_trigger'] | df['sell_trigger']

    # Initialize variables
    signal_indices = df.index[df['trade_signal']].tolist()
    accepted_indices = []
    last_accepted_time = None

    # Iterate over the indices where 'trade_signal' is True
    for idx in signal_indices:
        current_time = df.loc[idx, 'datetime']
        if last_accepted_time is None or (current_time - last_accepted_time).total_seconds() >= params['timer']:
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
    del fast_moving, is_slowing_down, state_array, compute_state, signal_indices
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

    entry_point_long = (df['buy_signal']) & (df['position'] != 0)
    entry_point_short = ( df['sell_signal']) & (df['position'] != 0)
    

    df.loc[entry_point_long, 'entry_price'] = df['Ask']
    df.loc[entry_point_short, 'entry_price'] = df['Bid']

    df['entry_price'] = df['entry_price'].where(df['position'] != 0).ffill()

    df['SL_price_long'] = df['entry_price'] - stop_loss
    df['TP_price_long'] = df['entry_price'] + take_profit
    df['SL_price_short'] = df['entry_price'] + stop_loss
    df['TP_price_short'] = df['entry_price'] - take_profit

    df['trade_id'] = df['buy_signal'].cumsum() + df['sell_signal'].cumsum()
    df['trade_id'] = df['trade_id'].where(df['position'] != 0)

    df['time_diff'] = df['datetime'].diff().dt.total_seconds().fillna(0)
    df['time_elapsed'] = df.groupby('trade_id')['time_diff'].cumsum()

# exit conditions for long and short trades making stop losses and take profit be limit orders based off bid/ask

    exit_short_stop = (df['position'] == -1) & (df['Last'] >= df['SL_price_short']) & (df['Volume'].notna())
    exit_short_take = (df['position'] == -1) & (df['Ask'] <= df['TP_price_short']) & (df['Volume'].notna())

    exit_long_stop = (df['position'] == 1) & (df['Last'] <= df['SL_price_long']) & (df['Volume'].notna())
    exit_long_take = (df['position'] == 1) & (df['Bid'] >= df['TP_price_long']) & (df['Volume'].notna())

    exit_due_to_time = (df['time_elapsed'] >= params['timer']) & (df['position'] != 0) & (df['Volume'].notna())

    # drop intermediate columns
    df.drop(['SL_price_long', 'TP_price_long', 'SL_price_short', 'TP_price_short'], axis=1, inplace=True)
    del take_profit, stop_loss, entry_point_long, entry_point_short, 
# finished here for the day. make a different price condition for each of these... with flags?

    exit_condition = (exit_short_stop| exit_short_take | exit_long_stop | exit_long_take | exit_due_to_time).fillna(False).astype(bool)
    df['exit_condition'] = exit_condition  # Store for reference

    cummax_exit_condition_int = df.groupby('trade_id')['exit_condition'].cummax()

    cummax_exit_condition = cummax_exit_condition_int.astype(bool)

    shifted_cummax = cummax_exit_condition.shift(fill_value=False)
    not_shifted_cummax = ~shifted_cummax



    # Exit due to time elapsed
    df['exit_flag_time'] = exit_due_to_time & not_shifted_cummax
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])
    df.loc[df['exit_flag_time'], 'exit_price'] = df.loc[df['exit_flag_time'], 'Bid'].where(df.loc[df['exit_flag_time'], 'position'] == 1, df.loc[df['exit_flag_time'], 'Ask'])

    # Exit due to stop loss for Long trades 
    df['exit_long_flag_stop'] = exit_long_stop & not_shifted_cummax
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])
    df.loc[df['exit_long_flag_stop'], 'exit_price'] = df.loc[df['exit_long_flag_stop'], 'Bid']

    # Exit due to stop loss for Short trades
    df['exit_short_flag_stop'] = exit_short_stop & not_shifted_cummax   
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])
    df.loc[df['exit_short_flag_stop'], 'exit_price'] = df.loc[df['exit_short_flag_stop'], 'Ask']

    # Exit due to take profit for Long trades
    df['exit_long_flag_take'] = exit_long_take & not_shifted_cummax
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])
    df.loc[df['exit_long_flag_take'], 'exit_price'] = df.loc[df['exit_long_flag_take'], 'Bid']

    # Exit due to take profit for Short trades
    df['exit_short_flag_take'] = exit_short_take & not_shifted_cummax
    df['time_elapsed'] = np.where(df['sell_signal'] | df['buy_signal'], 0, df['time_elapsed'])
    df.loc[df['exit_short_flag_take'], 'exit_price'] = df.loc[df['exit_short_flag_take'], 'Ask']

    df['exit_flag'] = df['exit_long_flag_stop'] | df['exit_long_flag_take'] | df['exit_short_flag_stop'] | df['exit_short_flag_take'] | df['exit_flag_time']
    df.drop(['exit_long_flag_stop', 'exit_long_flag_take', 'exit_short_flag_stop', 'exit_short_flag_take', 'exit_flag_time'], axis=1, inplace=True)

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
