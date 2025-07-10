import numpy as np
import pandas as pd
import time
from numba import njit


def calculate_trade_metrics(trades_data,params):
    # Precompute necessary boolean masks
    profit_positive = trades_data['profit_loss'] > 0
    profit_negative = trades_data['profit_loss'] < 0
    profit_zero = trades_data['profit_loss'] == 0
    exit_flag = trades_data['exit_flag'] == True
    position_long = trades_data['position'] == 1
    position_short = trades_data['position'] == -1
    time_elapsed_ge_120 = trades_data['time_elapsed'] >= 120

    if 'session_id' in trades_data.columns:
        trades_data['unique_trade_id'] = list(zip(trades_data['session_id'], trades_data['trade_id']))
    else:
        trades_data['unique_trade_id'] = trades_data['trade_id']

    # Calculate unique trade IDs once
    unique_unique_trade_ids = trades_data['unique_trade_id'].unique()
    # Total trades
    total_trades = len(unique_unique_trade_ids) -1
    # Winning trades
    winning_unique_trade_ids = trades_data.loc[profit_positive, 'unique_trade_id'].unique()
    winning_trades = len(winning_unique_trade_ids)
    # Losing trades
    losing_unique_trade_ids = trades_data.loc[profit_negative, 'unique_trade_id'].unique()
    losing_trades = len(losing_unique_trade_ids)
    # Total zero-point trades
    zero_point_unique_trade_ids = trades_data.loc[exit_flag & profit_zero, 'unique_trade_id'].unique()
    total_zero_point_trades = len(zero_point_unique_trade_ids)
    # Long trades
    long_unique_trade_ids = trades_data.loc[position_long, 'unique_trade_id'].unique()
    long_trades = len(long_unique_trade_ids)
    # Short trades
    short_unique_trade_ids = trades_data.loc[position_short, 'unique_trade_id'].unique()
    short_trades = len(short_unique_trade_ids)
    # Winning long trades
    winning_long_unique_trade_ids = trades_data.loc[position_long & profit_positive, 'unique_trade_id'].unique()
    winning_long_trades = len(winning_long_unique_trade_ids)
    # Losing long trades
    losing_long_unique_trade_ids = trades_data.loc[position_long & profit_negative, 'unique_trade_id'].unique()
    losing_long_trades = len(losing_long_unique_trade_ids)
    # Winning short trades
    winning_short_unique_trade_ids = trades_data.loc[position_short & profit_positive, 'unique_trade_id'].unique()
    winning_short_trades = len(winning_short_unique_trade_ids)
    # Losing short trades
    losing_short_unique_trade_ids = trades_data.loc[position_short & profit_negative, 'unique_trade_id'].unique()
    losing_short_trades = len(losing_short_unique_trade_ids)
    # Total profit/loss
    total_profit_loss = trades_data['profit_loss'].sum()
    #Total winning trade profit
    total_winning_trade_profit = trades_data.loc[profit_positive, 'profit_loss'].sum()
    # Total losing trade loss
    total_losing_trade_loss = trades_data.loc[profit_negative, 'profit_loss'].sum()
    # Total commissions
    total_commissions = total_trades * 1.04  # Assuming $1.04 commission per trade
    # Total time cut trades
    time_cut_unique_trade_ids = trades_data.loc[exit_flag & time_elapsed_ge_120, 'unique_trade_id'].unique()
    total_time_cut_trades = len(time_cut_unique_trade_ids)

    #most prof(loss) single day
    bestday = trades_data.groupby('date_ct')['profit_loss'].sum().max()
    # worst profit(loss) single day
    worstday = trades_data.groupby('date_ct')['profit_loss'].sum().min()
    # number of days that were profitable
    profitable_days = trades_data.groupby('date_ct')['profit_loss'].sum().gt(0).sum()
    # number of days that were not profitable
    unprofitable_days = trades_data.groupby('date_ct')['profit_loss'].sum().lt(0).sum()
    # daily profit
    # Group by date and sum the profit/loss for each day
    daily_profits = trades_data.groupby('date_ct')['profit_loss'].sum()

    # Initialize an empty list to store daily profits
    profits = []

    # Iterate over the number of days available in the data
    for day in range(len(daily_profits)):
        # Append the profit for the current day
        profits.append(daily_profits.iloc[day])

    # Win percentage
    win_percent = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    # Win Margin
    win_margin = (total_winning_trade_profit / (total_winning_trade_profit + abs(total_losing_trade_loss))) * 100  if (total_winning_trade_profit + abs(total_losing_trade_loss)) != 0 else 0
    # Win Margin Required
    win_required = (((params['stop_loss']*5) + 1.04) / (((params['stop_loss']*5) + 1.04) + ((params['take_profit']*5) - 1.04))) * 100

    daily_profits = trades_data.groupby('date_ct')['profit_loss'].sum()


    
    largest_winner = trades_data.loc[profit_positive, 'profit_loss'].max()
    largest_loser = trades_data.loc[profit_negative, 'profit_loss'].min()
    longest_trade_when_exit = trades_data.loc[exit_flag, 'time_elapsed'].max()
    shortest_trade_when_exit = trades_data.loc[exit_flag, 'time_elapsed'].min()
    most_ticks = trades_data['ticks'].max()

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
        'Total Winning Trade Profit': total_winning_trade_profit,
        'Total Losing Trade Loss': total_losing_trade_loss,
        'Total Commissions': total_commissions,
        'Total Time Cut Trades': total_time_cut_trades,
        'Best Day': bestday,
        'Worst Day': worstday,
        'Profitable Days': profitable_days,
        'Unprofitable Days': unprofitable_days,
        
        'Win Percentage': win_percent,
        'Win Margin': win_margin,
        'Win Margin Required': win_required,
        'Largest Winner': largest_winner,
        'Largest Loser': largest_loser,
        'Longest Trade when Exit': longest_trade_when_exit,
        'Shortest Trade when Exit': shortest_trade_when_exit,
        'Most Ticks': most_ticks

    }
    # Add daily profits to the metrics dictionary
    for date, profit in daily_profits.items():
        metrics[f'Profit on {date}'] = profit

    # Trades during 0:00 - 1:00
    trades_0_1 = trades_data[(trades_data['datetime'].dt.hour == 0) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_0_1 = trades_data[(trades_data['datetime'].dt.hour == 0) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_0_1 = trades_data[(trades_data['datetime'].dt.hour == 0) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_0_1 = trades_data[(trades_data['datetime'].dt.hour == 0) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 1:00 - 2:00
    trades_1_2 = trades_data[(trades_data['datetime'].dt.hour == 1) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_1_2 = trades_data[(trades_data['datetime'].dt.hour == 1) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_1_2 = trades_data[(trades_data['datetime'].dt.hour == 1) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_1_2 = trades_data[(trades_data['datetime'].dt.hour == 1) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 2:00 - 3:00
    trades_2_3 = trades_data[(trades_data['datetime'].dt.hour == 2) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_2_3 = trades_data[(trades_data['datetime'].dt.hour == 2) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_2_3 = trades_data[(trades_data['datetime'].dt.hour == 2) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_2_3 = trades_data[(trades_data['datetime'].dt.hour == 2) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 3:00 - 4:00
    trades_3_4 = trades_data[(trades_data['datetime'].dt.hour == 3) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_3_4 = trades_data[(trades_data['datetime'].dt.hour == 3) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_3_4 = trades_data[(trades_data['datetime'].dt.hour == 3) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_3_4 = trades_data[(trades_data['datetime'].dt.hour == 3) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 4:00 - 5:00
    trades_4_5 = trades_data[(trades_data['datetime'].dt.hour == 4) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_4_5 = trades_data[(trades_data['datetime'].dt.hour == 4) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_4_5 = trades_data[(trades_data['datetime'].dt.hour == 4) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_4_5 = trades_data[(trades_data['datetime'].dt.hour == 4) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 5:00 - 6:00
    trades_5_6 = trades_data[(trades_data['datetime'].dt.hour == 5) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_5_6 = trades_data[(trades_data['datetime'].dt.hour == 5) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_5_6 = trades_data[(trades_data['datetime'].dt.hour == 5) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_5_6 = trades_data[(trades_data['datetime'].dt.hour == 5) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 6:00 - 7:00
    trades_6_7 = trades_data[(trades_data['datetime'].dt.hour == 6) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_6_7 = trades_data[(trades_data['datetime'].dt.hour == 6) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_6_7 = trades_data[(trades_data['datetime'].dt.hour == 6) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_6_7 = trades_data[(trades_data['datetime'].dt.hour == 6) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 7:00 - 8:00
    trades_7_8 = trades_data[(trades_data['datetime'].dt.hour == 7) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_7_8 = trades_data[(trades_data['datetime'].dt.hour == 7) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_7_8 = trades_data[(trades_data['datetime'].dt.hour == 7) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_7_8 = trades_data[(trades_data['datetime'].dt.hour == 7) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 8:00 - 9:00
    trades_8_9 = trades_data[(trades_data['datetime'].dt.hour == 8) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_8_9 = trades_data[(trades_data['datetime'].dt.hour == 8) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_8_9 = trades_data[(trades_data['datetime'].dt.hour == 8) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_8_9 = trades_data[(trades_data['datetime'].dt.hour == 8) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 9:00 - 10:00
    trades_9_10 = trades_data[(trades_data['datetime'].dt.hour == 9) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_9_10 = trades_data[(trades_data['datetime'].dt.hour == 9) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_9_10 = trades_data[(trades_data['datetime'].dt.hour == 9) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_9_10 = trades_data[(trades_data['datetime'].dt.hour == 9) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 10:00 - 11:00
    trades_10_11 = trades_data[(trades_data['datetime'].dt.hour == 10) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_10_11 = trades_data[(trades_data['datetime'].dt.hour == 10) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_10_11 = trades_data[(trades_data['datetime'].dt.hour == 10) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_10_11 = trades_data[(trades_data['datetime'].dt.hour == 10) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 11:00 - 12:00
    trades_11_12 = trades_data[(trades_data['datetime'].dt.hour == 11) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_11_12 = trades_data[(trades_data['datetime'].dt.hour == 11) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_11_12 = trades_data[(trades_data['datetime'].dt.hour == 11) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_11_12 = trades_data[(trades_data['datetime'].dt.hour == 11) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 12:00 - 13:00
    trades_12_13 = trades_data[(trades_data['datetime'].dt.hour == 12) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_12_13 = trades_data[(trades_data['datetime'].dt.hour == 12) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_12_13 = trades_data[(trades_data['datetime'].dt.hour == 12) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_12_13 = trades_data[(trades_data['datetime'].dt.hour == 12) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 13:00 - 14:00
    trades_13_14 = trades_data[(trades_data['datetime'].dt.hour == 13) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_13_14 = trades_data[(trades_data['datetime'].dt.hour == 13) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_13_14 = trades_data[(trades_data['datetime'].dt.hour == 13) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_13_14 = trades_data[(trades_data['datetime'].dt.hour == 13) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 14:00 - 15:00
    trades_14_15 = trades_data[(trades_data['datetime'].dt.hour == 14) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_14_15 = trades_data[(trades_data['datetime'].dt.hour == 14) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_14_15 = trades_data[(trades_data['datetime'].dt.hour == 14) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_14_15 = trades_data[(trades_data['datetime'].dt.hour == 14) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 15:00 - 16:00
    trades_15_16 = trades_data[(trades_data['datetime'].dt.hour == 15) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_15_16 = trades_data[(trades_data['datetime'].dt.hour == 15) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_15_16 = trades_data[(trades_data['datetime'].dt.hour == 15) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_15_16 = trades_data[(trades_data['datetime'].dt.hour == 15) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 16:00 - 17:00
    trades_16_17 = trades_data[(trades_data['datetime'].dt.hour == 16) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_16_17 = trades_data[(trades_data['datetime'].dt.hour == 16) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_16_17 = trades_data[(trades_data['datetime'].dt.hour == 16) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_16_17 = trades_data[(trades_data['datetime'].dt.hour == 16) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 17:00 - 18:00
    trades_17_18 = trades_data[(trades_data['datetime'].dt.hour == 17) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_17_18 = trades_data[(trades_data['datetime'].dt.hour == 17) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_17_18 = trades_data[(trades_data['datetime'].dt.hour == 17) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_17_18 = trades_data[(trades_data['datetime'].dt.hour == 17) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 18:00 - 19:00
    trades_18_19 = trades_data[(trades_data['datetime'].dt.hour == 18) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_18_19 = trades_data[(trades_data['datetime'].dt.hour == 18) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_18_19 = trades_data[(trades_data['datetime'].dt.hour == 18) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_18_19 = trades_data[(trades_data['datetime'].dt.hour == 18) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 19:00 - 20:00
    trades_19_20 = trades_data[(trades_data['datetime'].dt.hour == 19) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_19_20 = trades_data[(trades_data['datetime'].dt.hour == 19) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_19_20 = trades_data[(trades_data['datetime'].dt.hour == 19) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_19_20 = trades_data[(trades_data['datetime'].dt.hour == 19) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 20:00 - 21:00
    trades_20_21 = trades_data[(trades_data['datetime'].dt.hour == 20) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_20_21 = trades_data[(trades_data['datetime'].dt.hour == 20) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_20_21 = trades_data[(trades_data['datetime'].dt.hour == 20) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_20_21 = trades_data[(trades_data['datetime'].dt.hour == 20) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 21:00 - 22:00
    trades_21_22 = trades_data[(trades_data['datetime'].dt.hour == 21) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_21_22 = trades_data[(trades_data['datetime'].dt.hour == 21) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_21_22 = trades_data[(trades_data['datetime'].dt.hour == 21) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_21_22 = trades_data[(trades_data['datetime'].dt.hour == 21) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 22:00 - 23:00
    trades_22_23 = trades_data[(trades_data['datetime'].dt.hour == 22) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_22_23 = trades_data[(trades_data['datetime'].dt.hour == 22) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_22_23 = trades_data[(trades_data['datetime'].dt.hour == 22) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_22_23 = trades_data[(trades_data['datetime'].dt.hour == 22) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()
    # trades during 23:00 - 24:00
    trades_23_24 = trades_data[(trades_data['datetime'].dt.hour == 23) & (trades_data['profit_loss'] != 0)]['unique_trade_id'].unique().tolist()
    wins_23_24 = trades_data[(trades_data['datetime'].dt.hour == 23) & (trades_data['profit_loss'] > 0)]['unique_trade_id'].unique().tolist()
    losses_23_24 = trades_data[(trades_data['datetime'].dt.hour == 23) & (trades_data['profit_loss'] < 0)]['unique_trade_id'].unique().tolist()
    time_cuts_23_24 = trades_data[(trades_data['datetime'].dt.hour == 23) & (trades_data['time_elapsed'] >= 120)]['unique_trade_id'].unique().tolist()

    time_metrics = {
        'trades': [total_trades, len(trades_22_23), len(trades_23_24), len(trades_0_1), len(trades_1_2), len(trades_2_3), len(trades_3_4), len(trades_4_5), len(trades_5_6),
                    len(trades_6_7), len(trades_7_8), len(trades_8_9), len(trades_9_10), len(trades_10_11), len(trades_11_12),
                    len(trades_12_13), len(trades_13_14), len(trades_14_15), len(trades_15_16), len(trades_16_17), len(trades_17_18),
                    len(trades_18_19), len(trades_19_20), len(trades_20_21), len(trades_21_22)],

        'wins': [winning_trades,  len(wins_22_23), len(wins_23_24), len(wins_0_1), len(wins_1_2), len(wins_2_3), len(wins_3_4), len(wins_4_5), len(wins_5_6), len(wins_6_7), 
                 len(wins_7_8), len(wins_8_9), len(wins_9_10), len(wins_10_11), len(wins_11_12), len(wins_12_13), 
                 len(wins_13_14), len(wins_14_15), len(wins_15_16), len(wins_16_17), len(wins_17_18), len(wins_18_19), 
                 len(wins_19_20), len(wins_20_21), len(wins_21_22)],
                 
        'losses': [losing_trades, len(losses_22_23), len(losses_23_24), len(losses_0_1), len(losses_1_2), len(losses_2_3), len(losses_3_4), len(losses_4_5), len(losses_5_6), 
                   len(losses_6_7), len(losses_7_8), len(losses_8_9), len(losses_9_10), len(losses_10_11), len(losses_11_12), 
                   len(losses_12_13), len(losses_13_14), len(losses_14_15), len(losses_15_16), len(losses_16_17), len(losses_17_18), 
                   len(losses_18_19), len(losses_19_20), len(losses_20_21), len(losses_21_22)],

        'time_cuts': [total_time_cut_trades, len(time_cuts_22_23), len(time_cuts_23_24), len(time_cuts_0_1), len(time_cuts_1_2), len(time_cuts_2_3), len(time_cuts_3_4), len(time_cuts_4_5), 
                      len(time_cuts_5_6), len(time_cuts_6_7), len(time_cuts_7_8), len(time_cuts_8_9), len(time_cuts_9_10), 
                      len(time_cuts_10_11), len(time_cuts_11_12), len(time_cuts_12_13), len(time_cuts_13_14), len(time_cuts_14_15), 
                      len(time_cuts_15_16), len(time_cuts_16_17), len(time_cuts_17_18), len(time_cuts_18_19), len(time_cuts_19_20), 
                      len(time_cuts_20_21), len(time_cuts_21_22)]     
    }

    return metrics, time_metrics


