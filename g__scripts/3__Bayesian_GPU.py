# ---------------------------------------------------------------------------------
# -------------- 3. Bayesian Optimization for Trade Strategy ---------------------
# ---------------------------------------------------------------------------------

# Standard library imports
import sys
import os
import time
import logging
import copy
import gc
import traceback
from glob import glob

# Third-party imports
import yaml
import pandas as pd
import numpy as np
import optuna
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

from optuna import Trial
from optuna.samplers import TPESampler
from optuna.storages import RDBStorage
from optuna.visualization import plot_optimization_history, plot_param_importances

# Local application imports
sys.path.append('/home/cheddarjackk/Developer/StrategyBuilder/')
from f__modules.strat import process_tick_data
from f__modules.dat_handler import load_data
from f__modules.trades_simulate import generate_signals, simulate_trades, extract_trades
from f__modules.metrics import calculate_trade_metrics

# ---------------------------------------------------------------------------------
# Verify that TensorFlow sees the GPU
# ---------------------------------------------------------------------------------
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Restrict TensorFlow to only use the first GPU
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is set for TensorFlow.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected by TensorFlow. Proceeding on CPU.")


# ---------------------------------------------------------------------------------
# Constants / Config
# ---------------------------------------------------------------------------------
TRIALS = 100
CONFIG_PATH = '/home/cheddarjackk/Developer/StrategyBuilder/c__config/config.yaml'
DATA_DIR = '/home/cheddarjackk/Developer/StrategyBuilder/d__data/3_day_data_preprocessed/'
STUDY_NAME = 'trade_strategy_optimization'

# MariaDB connection URL
mariadb_password = os.getenv('MARIADB_PASSWORD')  # export MARIADB_PASSWORD=your_password_here
if not mariadb_password:
    raise ValueError("Please set the MARIADB_PASSWORD environment variable.")
mariadb_url = f'mariadb+pymysql://optuna_user:{mariadb_password}@localhost:3306/optuna_database'

# Configure logging
logging.basicConfig(
    filename='StrategyBuilder/e__output/optimization.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)
optuna.logging.set_verbosity(logging.WARNING)

# ---------------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def convert_time(time_in_seconds):
    from datetime import datetime, timedelta
    import pytz
    central_tz = pytz.timezone('US/Central')
    # Convert seconds since midnight UTC to datetime
    utc_dt = datetime(1970, 1, 1, tzinfo=pytz.utc) + timedelta(seconds=time_in_seconds)
    # Convert to Central Time
    central_dt = utc_dt.astimezone(central_tz)
    return central_dt.strftime('%I:%M %p')  # a.m./p.m. format

def custom_callback(total_TRIALS):
    def callback(study, trial):
        trades = trial.user_attrs.get('trades', np.nan)
        profit = trial.user_attrs.get('profit', np.nan)
        duration = trial.user_attrs.get('duration', 0)
        margin = trial.user_attrs.get('margin', np.nan)
        sessions = trial.user_attrs.get('sessions', 0)
        adjustment = trial.user_attrs.get('adjustment', np.nan)

        def safe_format(value, fmt=".2f", suffix=""):
            if isinstance(value, (float, int)) and not np.isnan(value):
                return f"{value:{fmt}}{suffix}"
            return "N/A"

        margin_str = safe_format(margin, suffix="%")
        profit_str = safe_format(profit)
        adjustment_str = safe_format(adjustment)
        duration_str = safe_format(duration, fmt=".0f", suffix=" seconds")
        value = safe_format(trial.value)

        session_prefix_mapping = {
            1:  "1__________ - ",
            2:  "_2_________ - ",
            3:  "__3________ - ",
            4:  "___4_______ - ",
            5:  "____5______ - ",
            6:  "_____6_____ - ",
            7:  "______7____ - ",
            8:  "_______8___ - ",
            9:  "________9__ - ",
            10: "_________10 - ",
            11: "_________11 - ",
            12: "_________12 - ",
            13: "_________13 - ",
            14: "_________14 - ",
            15: "_________15 - ",
            16: "_________16 - ",
            17: "_________17 - ",
            18: "_________18 - ",
            19: "_________19 - ",
            20: "_________20 - ",
        }

        if sessions in session_prefix_mapping and profit > 0:
            prefix = session_prefix_mapping[sessions]
            print(f"{prefix}{trial.number} | {margin_str} | $$$$$ {profit_str} "
                  f"| adj: {adjustment_str} | trades: {trades} | score: {value} "
                  f"| [{duration_str}]")
        elif sessions in session_prefix_mapping and profit < 0:
            prefix = session_prefix_mapping[sessions]
            print(f"{prefix}{trial.number} | {margin_str} | ----$ {profit_str} "
                  f"| adj: {adjustment_str} | trades: {trades} | score: {value} "
                  f"| [{duration_str}]")
        else:
            print(f"Trial {trial.number} | sessions: {sessions} | {margin_str} | "
                  f"$$$: {profit_str} | adj: {adjustment_str} | trades: {trades} "
                  f"| score: {value} | [{duration_str}]")
    return callback

# ---------------------------------------------------------------------------------
# The Function That Does GPU Work on a Single File
# ---------------------------------------------------------------------------------

def process_single_session(file_path, config):
    # 1) Load tick data
    tick_data = load_data(file_path)

    # 2) Separate 'datetime' and numeric columns
    if 'datetime' not in tick_data.columns:
        raise KeyError("'datetime' column not found in tick_data.")
    
    datetime_data = tick_data['datetime'].reset_index(drop=True)
    numeric_cols = tick_data.select_dtypes(include=[np.number]).columns
    numeric_data = tick_data[numeric_cols]

    # 3) GPU-accelerated TF transforms
    with tf.device('/GPU:0'):
        tick_tensor = tf.convert_to_tensor(numeric_data.values, dtype=tf.float32)
        mean = tf.reduce_mean(tick_tensor, axis=0)
        std_dev = tf.math.reduce_std(tick_tensor, axis=0)
        normalized_tensor = (tick_tensor - mean) / (std_dev + 1e-8)
        
        dense = tf.keras.layers.Dense(units=len(numeric_cols), activation='relu')
        processed_tensor = dense(normalized_tensor)

        # Convert to DataFrame (with the same numeric columns)
        processed_data_numeric = pd.DataFrame(processed_tensor.numpy(), columns=numeric_cols)
    
    # 4) Combine 'datetime' with processed numeric data
    processed_data = pd.concat([datetime_data, processed_data_numeric], axis=1)
        
    # 5) Generate signals and simulate trades
    processed_data = generate_signals(processed_data, config['parameters'])
    processed_data = simulate_trades(processed_data, config['parameters'])
    trades_data = extract_trades(processed_data)

    del processed_data
    gc.collect()
    return trades_data

# ---------------------------------------------------------------------------------
# The Optuna Objective
# ---------------------------------------------------------------------------------
def objective(trial: Trial):
    start_time = time.time()
    config = load_config(CONFIG_PATH)

    # Suggest hyperparameters
    candles = trial.suggest_int('candles', 3, 30)
    k_length = trial.suggest_int('k_length', 5, 30)
    Slow_K_High = trial.suggest_int('Slow_K_High', 70, 95)
    Slow_K_High_cap_min = max(Slow_K_High + 5, 70)
    Slow_K_High_cap = trial.suggest_int('Slow_K_High_cap', Slow_K_High_cap_min, 100)
    Slow_K_Low = trial.suggest_int('Slow_K_Low', 5, 30)
    Slow_K_Low_cap_min = min(Slow_K_Low - 5, 30)
    Slow_K_Low_cap = trial.suggest_int('Slow_K_Low_cap', 0, Slow_K_Low_cap_min)
    floor = trial.suggest_int('min_value', 0, 150, step=10)
    spacing_1 = max(floor + 50, 100)
    bottom_bunk = trial.suggest_int('high_value', spacing_1, 350, step=10)
    spacing_2 = max(bottom_bunk + 50, 400) 
    top_bunk = trial.suggest_int('low_value', spacing_2, 900, step=10)
    spacing_3 = max(top_bunk + 150, 500)
    ceiling = trial.suggest_int('max_value', spacing_3, 1500, step=10)
    kids = trial.suggest_float('low_average', 1.00, 4.00, step=0.25)
    age_1 = max(kids + 2.50, 4.00)
    parents = trial.suggest_float('high_average', age_1, 10.00, step=0.25)
    age_2 = max(parents + 2.00, 8.00)
    retirement = trial.suggest_float('threshold', age_2, 15.00, step=0.25)
    take_profit = trial.suggest_float('take_profit', 0.75, 4.0, step=0.25)
    stop_loss = trial.suggest_float('stop_loss', 0.75, 4.0, step=0.25)
    RoC_threshold = trial.suggest_int('RoC_threshold', 40, 250, step=10)
    timer = trial.suggest_int('timer', 60, 300, step=10)

    params = {
        'k_length': k_length,
        'Slow_K_High': Slow_K_High, 'Slow_K_High_cap': Slow_K_High_cap,
        'Slow_K_Low': Slow_K_Low, 'Slow_K_Low_cap': Slow_K_Low_cap,
        'take_profit': take_profit, 'stop_loss': stop_loss,
        'candles': candles, 'high_average': parents, 'low_average': kids,
        'threshold': retirement, 'high_value': bottom_bunk, 'low_value': top_bunk,
        'min_value': floor, 'max_value': ceiling,
        'RoC_threshold': RoC_threshold, 'timer': timer
    }
    config['parameters'].update(params)

    try:
        # ------------------------------------------------------------
        # Example quick GPU test in the objective (optional)
        # ------------------------------------------------------------
        with tf.device('/GPU:0'):
            dummy_tensor = tf.convert_to_tensor(np.random.random((1000, 1000)), dtype=tf.float32)
            matmul_res = tf.matmul(dummy_tensor, dummy_tensor)
            result = tf.reduce_sum(matmul_res)
        print(f"TensorFlow GPU test result: {result.numpy()}")

        # ------------------------------------------------------------
        # Sessions loop (single process)
        # ------------------------------------------------------------
        session_ranges = [
            (170, 180), (160, 169), (150, 159), (140, 149), (130, 139),
            (120, 129), (110, 119), (100, 109), (90, 99),  (80, 89),
            (70, 79),   (60, 69),   (50, 59),   (40, 49),  (30, 39),
            (20, 29),   (10, 19),   (0, 9)
        ]

        total_profit = 0
        total_win_margin = 0
        total_trades = 0
        total_sessions = 0
        win_margin_added = 0

        session_profits = []
        session_margin = []
        session_trades = []

        # We do NOT use multiprocessing pool here.
        for session_range in session_ranges:
            total_sessions += 1
            start_session_id, end_session_id = session_range

            # Gather the parquet files for this session range
            all_files = glob(os.path.join(DATA_DIR, 'session_*.parquet'))
            selected_files = []
            for file in all_files:
                file_name = os.path.basename(file)
                parts = file_name.replace('.parquet', '').split('_')
                try:
                    session_id = int(float(parts[1]))
                except (IndexError, ValueError):
                    continue

                if start_session_id <= session_id <= end_session_id:
                    selected_files.append(file)

            selected_files.sort(
                key=lambda x: int(float(os.path.basename(x).split('_')[1]))
            )

            # Process each file in this session (on GPU) sequentially
            all_trades_data = []
            for file in selected_files:
                trades_data = process_single_session(file, config)
                all_trades_data.append(trades_data)

            # Combine trades
            if all_trades_data:
                trades_data_combined = pd.concat(all_trades_data, ignore_index=True)
            else:
                trades_data_combined = pd.DataFrame()

            # Calculate metrics
            metrics, _ = calculate_trade_metrics(trades_data_combined, config['parameters'])
            win_margin = metrics['Win Margin']
            last_profit = metrics['Total Profit/Loss']
            last_trades = metrics['Total Trades']

            session_trades.append(last_trades)
            session_profits.append(last_profit)
            session_margin.append(win_margin)

            total_trades += last_trades
            win_margin_added += win_margin
            total_win_margin = win_margin_added / total_sessions
            total_profit += last_profit

            # Basic early stop
            if total_win_margin < 47 or win_margin < 42 or last_trades < 4:
                break

            del trades_data_combined
            gc.collect()

        # ------------------------------------------------------------
        # Compute final "score"
        # ------------------------------------------------------------
        profit_weights = [3.0, 3.0, 3.0, 3.0, 2.0, 1.5, 1.0]
        sessions_weights = [600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0]
        default_weight = 2700.0
        combined_score = 0.0

        if session_profits and session_profits[0] > 0:
            for i, profit in enumerate(session_profits):
                w = profit_weights[i] if i < len(profit_weights) else profit_weights[-1]
                combined_score += profit * w

            if total_sessions <= len(sessions_weights):
                combined_score += sessions_weights[total_sessions - 1]
            else:
                combined_score += default_weight

            # Slight extra weighting
            combined_score += (total_profit - sum(session_profits[:5]))  
        else:
            # Fallback if no initial profit
            combined_score = (total_profit * profit_weights[0]) - sessions_weights[0]

        # Adjust combined_score based on total_win_margin
        k = 8.15
        n = 1.2
        o = abs((total_win_margin - 50) / 100)
        p = k * (o ** n)
        adj_margin = p

        if total_win_margin > 50 and total_sessions > 1:
            combined_score = combined_score + (combined_score * p)
        elif total_win_margin < 50 and total_sessions > 1:
            combined_score = combined_score - (combined_score * p)
        else:
            combined_score = combined_score * p

        # Set trial attributes
        trial.set_user_attr('profit', float(total_profit))
        trial.set_user_attr('trades', int(total_trades))
        trial.set_user_attr('margin', float(total_win_margin))
        trial.set_user_attr('sessions', int(total_sessions))
        trial.set_user_attr('adjustment', float(adj_margin))

        # Exclude strategies with too few trades
        if total_trades <= 5:
            logging.info(f"Trial {trial.number}: Insufficient trades ({total_trades}). Parameters: {params}")
            return -400

        # Log
        elapsed_time = time.time() - start_time
        trial.set_user_attr('duration', elapsed_time)
        logging.info(f"Trial {trial.number}: Total Profit: {total_profit}, Total Trades: {total_trades}, "
                     f"Parameters: {params}")

        return combined_score

    except Exception as e:
        logging.error(f"Trial {trial.number}: Error with parameters {params}: {e}")
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logging.error(f"Full traceback:\n{traceback_str}")

        # Mark the trial as failed
        trial.set_user_attr('profit', float('nan'))
        trial.set_user_attr('trades', 0)
        trial.set_user_attr('margin', float('nan'))
        trial.set_user_attr('sessions', 0)
        trial.set_user_attr('adjustment', float('nan'))
        trial.set_user_attr('duration', float('nan'))
        return -1000

    finally:
        gc.collect()

# ---------------------------------------------------------------------------------
# Main: Create/Load Study and Optimize
# ---------------------------------------------------------------------------------
output_dir = 'output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

storage = RDBStorage(url=mariadb_url)

try:
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    logging.info("Loaded existing study.")
except KeyError:
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=storage,
        direction='maximize',
        sampler=TPESampler(seed=42)
    )
    logging.info("Created a new study.")

# IMPORTANT: Use n_jobs=1 to avoid multiple processes interfering with GPU usage
study.optimize(
    objective,
    n_trials=TRIALS,
    n_jobs=1,  # single process to avoid GPU conflicts
    callbacks=[custom_callback(TRIALS)]
)

# Print the best result
best_params = study.best_params
best_value = study.best_value
best_trial = study.best_trial

print("Best parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")

print(f"--- Adjusted Value: {best_value:.2f}")
print(f"--- Sessions Reached: {best_trial.user_attrs['sessions']}")
print(f"--- Total Trades: {best_trial.user_attrs['trades']}")
print(f"--- Avg Margin: {best_trial.user_attrs['margin']:.2f}%")
print(f"--- Total Profit: ${best_trial.user_attrs['profit']:,.2f}")

# Extract trial data
trials_data = []
for tr in study.trials:
    trial_dict = {
        'number': tr.number,
        'value': tr.value,
        'profit': tr.user_attrs.get('profit', np.nan),
        'trades': tr.user_attrs.get('trades', np.nan),
        'state': tr.state,
        'duration': tr.duration,
        'margin': tr.user_attrs.get('margin', np.nan),
        'sessions': tr.user_attrs.get('sessions', np.nan),
        'adjustment': tr.user_attrs.get('adjustment', np.nan),
    }
    # Include all parameters
    trial_dict.update(tr.params)
    trials_data.append(trial_dict)

results_df = pd.DataFrame(trials_data)

# Example reordering of columns (only if they exist)
desired_order = [
    'number','value','sessions','margin','profit','trades',
    'k_length','Slow_K_High','Slow_K_High_cap','Slow_K_Low','Slow_K_Low_cap',
    'candles','high_average','high_value','low_average','low_value','threshold', 
    'min_value','max_value','RoC_threshold','take_profit','stop_loss','timer',
    'state','duration'
]
existing_order = [col for col in desired_order if col in results_df.columns]
results_df = results_df[existing_order]

results_df.to_parquet(os.path.join(output_dir, 'results.parquet'))

# Plot optimization history
fig1 = plot_optimization_history(study)
fig1.write_image(os.path.join(output_dir, 'optimization_history.png'),
                 width=1600, height=900, scale=2)

# Plot parameter importances
fig2 = plot_param_importances(study)
fig2.write_image(os.path.join(output_dir, 'param_importances.png'),
                 width=1600, height=900, scale=2)

fig1.update_layout(
    font=dict(size=16),
    autosize=False,
    width=1920,
    height=1080,
)
fig1.write_html(os.path.join(output_dir, 'optimization_history.html'))
fig2.write_html(os.path.join(output_dir, 'param_importances.html'))

gc.collect()

# ----------------------------------------------------------------
# You can now run:
#   python your_script.py
# and everything will execute in a single process with GPU usage.
# ----------------------------------------------------------------
