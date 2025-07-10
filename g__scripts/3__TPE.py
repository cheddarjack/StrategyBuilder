# -----------------------------------------------------------------------------
# -------------- 3. Tree-structured Parzen Estimator (TPE) --------------------
# ----------------------------------------------------------------------------- 


# Standard library imports
import sys
import os
import time
import logging
import copy
import gc
import traceback
from glob import glob
from multiprocessing import Pool

# Third-party imports
import yaml
import pandas as pd
import numpy as np
import optuna
import psutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# Define P-Core PUs
P_CORE_PUS = list(range(16))  # P#0 to P#15
SUB_JOBS = 1
JOBS = 8
TRIALS = 1000
CONFIG_PATH = '/home/cheddarjackk/Developer/StrategyBuilder/c__config/config.yaml'
DATA_DIR = '/home/cheddarjackk/Developer/StrategyBuilder/d__data/3_day_data_preprocessed/'
STUDY_NAME = 'trade_strategy_optimization'
1
# MariaDB connection URL
mariadb_password = os.getenv('MARIADB_PASSWORD') #export MARIADB_PASSWORD=your_password_here
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

#----------------------------------------------------------------------#
#---------------------| Precess Functions |----------------------------#
#----------------------------------------------------------------------#

def set_cpu_affinity(pus):
    """Set CPU affinity for the current process to the specified PUs."""
    p = psutil.Process(os.getpid())
    p.cpu_affinity(pus)

def load_config(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as file:
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
    # Format time based on hour
    return central_dt.strftime('%I:%M %p')  # a.m./p.m. format
   
def custom_callback(total_TRIALS):
    def callback(study, trial):
        # Retrieve the number of trades from the trial's user attributes
        trades = trial.user_attrs.get('trades', np.nan)
        profit = trial.user_attrs.get('profit', np.nan)
        # start_time1 = trial.params.get('start_time1', 0)
        # end_time1 = trial.params.get('end_time1', 0)
        duration = trial.user_attrs.get('duration', 0)
        margin = trial.user_attrs.get('margin', np.nan)
        sessions = trial.user_attrs.get('sessions', 0)
        adjustment = trial.user_attrs.get('adjustment', np.nan)
    
        # try:
        #     start_time_formatted = convert_time(start_time1)
        #     end_time_formatted = convert_time(end_time1)
        # except Exception as e:
        #     start_time_formatted = "Invalid Time"
        #     end_time_formatted = "Invalid Time"
        #     logging.error(f"Trial {trial.number}: Time conversion error: {e}")
    
        # Safely format float attributes
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

        if sessions in session_prefix_mapping and profit > 0 :
            prefix = session_prefix_mapping[sessions]
            print(f"{prefix}{trial.number} | {margin_str} | $$$$$ {profit_str} | adj: {adjustment_str} | trades: {trades} | score: {value} | [{duration_str}]")
        elif sessions in session_prefix_mapping and profit < 0 :
            prefix = session_prefix_mapping[sessions]
            print(f"{prefix}{trial.number} | {margin_str} | ----$ {profit_str} | adj: {adjustment_str} | trades: {trades} | score: {value} | [{duration_str}]")
        else:
            print(f"Trial {trial.number} | sesh: {sessions} | {margin_str} | $: {profit_str} | adj: {adjustment_str} | trades: {trades} | score: {value} | [{duration_str}]")
    return callback

def process_single_session(file_path, config):
    # Load tick data
    tick_data = load_data(file_path)

    # Process tick data
    tick_data = process_tick_data(tick_data, config['parameters'])
    tick_data = generate_signals(tick_data, config['parameters'])
    tick_data = simulate_trades(tick_data, config['parameters'])

    trades_data = extract_trades(tick_data)

    # Free up memory
    del tick_data
    gc.collect()

    return trades_data

# main function to optimize the parameters
def objective(trial: Trial):
    # Start timer
    start_time = time.time()


    config = load_config(CONFIG_PATH)

    # Define the parameters with constraints using Optuna's suggest methods
    candles = trial.suggest_int('candles', 3, 30)

    k_length = trial.suggest_int('k_length', 5, 30)
    Slow_K_High = trial.suggest_int('Slow_K_High', 70, 95)
    Slow_K_High_cap_min = max(Slow_K_High + 5, 70)
    Slow_K_High_cap = trial.suggest_int('Slow_K_High_cap', 70, 100)
    Slow_K_Low = trial.suggest_int('Slow_K_Low', 5, 30)
    Slow_K_Low_cap_min = min(Slow_K_Low - 5, 30)
    Slow_K_Low_cap = trial.suggest_int('Slow_K_Low_cap', 0, 30)

    floor = trial.suggest_int('min_value', 0, 150, step=10)
    spacing_1 = max(floor + 50, 100)
    bottom_bunk = trial.suggest_int('high_value', 0, 350, step=10)
    spacing_2 = max(bottom_bunk + 50, 400) 
    top_bunk = trial.suggest_int('low_value', 350, 900, step=10)
    spacing_3 = max(top_bunk + 150, 500)
    ceiling = trial.suggest_int('max_value', 500, 1500, step=10)

    kids = trial.suggest_float('low_average', 1.00, 4.00, step=0.25)
    age_1 = max(kids + 2.50, 4.00)
    parents = trial.suggest_float('high_average', 3.00, 10.00, step=0.25)
    age_2 = max(parents + 2.00, 8.00)
    retirement = trial.suggest_float('threshold', 8.00, 15.00, step=0.25)

    take_profit = trial.suggest_float('take_profit', 0.75, 4.0, step=0.25)
    stop_loss = trial.suggest_float('stop_loss', 0.75, 4.0, step=0.25)
    RoC_threshold = trial.suggest_int('RoC_threshold', 40, 250, step=10)
    timer = trial.suggest_int ('timer', 60, 300, step=10)

    # candles = trial.suggest_int('candles', 3, 30)

    # k_length = trial.suggest_int('k_length', 5, 30)
    # Slow_K_High = trial.suggest_int('Slow_K_High', 70, 95)
    # Slow_K_High_cap_min = max(Slow_K_High + 5, 70)
    # Slow_K_High_cap = trial.suggest_int('Slow_K_High_cap', Slow_K_High_cap_min, 100)
    # Slow_K_Low = trial.suggest_int('Slow_K_Low', 5, 30)
    # Slow_K_Low_cap_min = min(Slow_K_Low - 5, 30)
    # Slow_K_Low_cap = trial.suggest_int('Slow_K_Low_cap', 0, Slow_K_Low_cap_min)

    # floor = trial.suggest_int('min_value', 0, 150, step=10)
    # spacing_1 = max(floor + 50, 100)
    # bottom_bunk = trial.suggest_int('high_value', spacing_1, 350, step=10)
    # spacing_2 = max(bottom_bunk + 50, 400) 
    # top_bunk = trial.suggest_int('low_value', spacing_2, 900, step=10)
    # spacing_3 = max(top_bunk + 150, 500)
    # ceiling = trial.suggest_int('max_value', spacing_3, 1500, step=10)
    
    # kids = trial.suggest_float('low_average', 1.00, 4.00, step=0.25)
    # age_1 = max(kids + 2.50, 4.00)
    # parents = trial.suggest_float('high_average', age_1, 10.00, step=0.25)
    # age_2 = max(parents + 2.00, 8.00)
    # retirement = trial.suggest_float('threshold', age_2, 15.00, step=0.25)

    # take_profit = trial.suggest_float('take_profit', 0.75, 4.0, step=0.25)
    # stop_loss = trial.suggest_float('stop_loss', 0.75, 4.0, step=0.25)
    # RoC_threshold = trial.suggest_int('RoC_threshold', 40, 250, step=10)
    # timer = trial.suggest_int ('timer', 60, 300, step=10)
    # NYSE in UTC time: 13:30:00 - 20:00:00
    # CME closed in UTC time: 21:00:00 - 22:00:00
    # start_time1 = trial.suggest_int('start_time1', 0, 86340, step=60) 
    # end_time1 = trial.suggest_int('end_time1', 0, 86340, step=60)

    try: 

        core_id = int(os.environ.get('PROCESS_CORE_ID', 0))
        set_cpu_affinity([P_CORE_PUS[core_id % len(P_CORE_PUS)]])

        params = {
        'k_length': k_length,
        'Slow_K_High': Slow_K_High, 'Slow_K_High_cap': Slow_K_High_cap,
        'Slow_K_Low': Slow_K_Low, 'Slow_K_Low_cap': Slow_K_Low_cap,
        'take_profit': take_profit, 'stop_loss': stop_loss,
        'candles': candles,'high_average': parents,'low_average': kids,'threshold': retirement,
        'high_value': bottom_bunk,'low_value': top_bunk, 'min_value': floor,'max_value': ceiling,
        'RoC_threshold': RoC_threshold,
        'timer': timer,
        # 'start_time1': start_time1, 'end_time1': end_time1,
        } 
        config['parameters'].update(params)

        session_ranges = [
            (170, 180), (160, 169), (150, 159), (140, 149), (130, 139),
            (120, 129), (110, 119), (100, 109), (90, 99), (80, 89),
            (70, 79), (60, 69), (50, 59), (40, 49), (30, 39),
            (20, 29), (10, 19), (0, 9)
        ]
        total_profit = 0
        total_win_margin = 0
        total_trades = 0
        total_sessions = 0
        win_margin_added = 0
        session_profits = []
        session_margin = []
        session_trades = []
        session_weights = []

        for session_range in session_ranges:
            total_sessions += 1
            start_session_id, end_session_id = session_range

            # Prepare the list of files for the current session range
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

            # Sort files by session ID
            selected_files.sort(key=lambda x: int(float(os.path.basename(x).split('_')[1])))
            config_for_multiprocessing = copy.deepcopy(config)

            num_workers = SUB_JOBS
            args = [(file, config_for_multiprocessing) for file in selected_files]

            with Pool(processes=num_workers, initializer=init_worker, initargs=(P_CORE_PUS,)) as pool:
                all_trades_data = pool.starmap(process_single_session, args)

            if all_trades_data:
                trades_data_combined = pd.concat(all_trades_data, ignore_index=True)
            else:
                trades_data_combined = pd.DataFrame()

            metrics, _ = calculate_trade_metrics(trades_data_combined, config['parameters'])

            # Calculate intermediate variables
            win_margin = metrics['Win Margin']
            win_margin_added += win_margin
            last_profit = metrics['Total Profit/Loss']
            last_trades = metrics['Total Trades']

            session_trades.append(last_trades)
            session_profits.append(last_profit)
            session_margin.append(win_margin)
            total_trades += metrics['Total Trades']
            total_win_margin = win_margin_added / total_sessions
            total_profit += last_profit

            # Check profitability threshold
            if total_win_margin < 50 or win_margin < 45 or last_trades < 5:
                break

            # Clear trades_data_combined to free memory
            del trades_data_combined
            gc.collect()

        # Define weights based on session number
        profit_weights = [3.0, 3.0, 3.0, 3.0, 2.0, 1.5, 1.0]  # For sessions 1 to 7
        sessions_weights = [600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0]
        default_weight = 2700.0
        combined_score = 0.0

        if session_profits[0] > 0:
            for i, profit in enumerate(session_profits):
                if i < len(profit_weights):
                    combined_score += profit * profit_weights[i]
                else:
                    combined_score += profit * profit_weights[-1]  # Use the last weight for additional sessions
                # print(f"trial number: {trial.number} session: {i} profit: {profit} margin: {session_margin[i]} trades: {session_trades[i]}")

            if total_sessions <= len(sessions_weights):
                combined_score += sessions_weights[total_sessions - 1]
            else:
                combined_score += default_weight

            combined_score += (total_profit - sum(session_profits[:5]))  # Adjust based on first five sessions
        else:
            combined_score = (total_profit * profit_weights[0]) - sessions_weights[0]


        # print(f"trial: {trial.number} score one: {combined_score} total profit: {total_profit} session weights: {sessions_weights[0]} profit weights: {profit_weights[0]}")

        # Adjust combined_score based on total_win_margin
        k = 8.15
        n = 1.2
        o = abs((total_win_margin - 50) / 100)
        p = k * (o ** n)
        adj_margin = p

        if total_win_margin > 50 and total_sessions > 1:
                combined_score = combined_score + (combined_score * (p * 1))
        elif total_win_margin < 50 and total_sessions > 1:
                combined_score = combined_score - (combined_score * (p * 1))
        else:
                combined_score = (combined_score * (p * 1))
        
        # print(f"trial: {trial.number} score two: {combined_score}")


        # Set trial attributes with float values
        trial.set_user_attr('profit', float(total_profit))
        trial.set_user_attr('trades', total_trades)
        trial.set_user_attr('margin', float(total_win_margin))
        trial.set_user_attr('sessions', total_sessions)
        trial.set_user_attr('adjustment', float(adj_margin))

        # Exclude strategies with too few trades
        if total_trades < 5:
            logging.info(f"Trial {trial.number}: Insufficient trades ({total_trades}). Parameters: {params}")
            return -2000

        # Log trial details
        elapsed_time = time.time() - start_time
        trial.set_user_attr('duration', elapsed_time)
        logging.info(f"Trial {trial.number}: Total Profit: {total_profit}, Total Trades: {total_trades}, Parameters: {params}")

        # Return the total profit for maximization
        return combined_score

    except Exception as e:
        logging.error(f"Trial {trial.number}: Error with parameters {params}: {e}")
        traceback_str = ''.join(traceback.format_exception(None, e, e.__traceback__))
        logging.error(f"Full traceback:\n{traceback_str}")
        trial.set_user_attr('profit', float('nan'))
        trial.set_user_attr('trades', 0)
        trial.set_user_attr('margin', float('nan'))
        trial.set_user_attr('sessions', 0)
        trial.set_user_attr('adjustment', float('nan'))
        trial.set_user_attr('duration', float('nan'))
        
        return -2000

    finally:
        gc.collect()

#----------------------------------------------------------------------#
#---------------------| Main Function |-------------------------------#
#----------------------------------------------------------------------#

def init_worker(pus):
    """Initialize worker by setting CPU affinity."""
    set_cpu_affinity(pus)

# Ensure the output directory exists
output_dir = 'e__output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define the storage to save the study's state (SQLite database)
storage = RDBStorage(url=mariadb_url)

sampler = TPESampler(
    n_startup_trials=50,
    n_ei_candidates=24,
    multivariate=True,
    group=True,
    seed=42
)

# Create or load the study
try:
    study = optuna.load_study(study_name=STUDY_NAME, storage=storage)
    logging.info("Loaded existing study.")
except KeyError:
    study = optuna.create_study(study_name=STUDY_NAME,storage=storage,direction='maximize',sampler=sampler)
    logging.info("Created a new study.")

# Run the optimization
study.optimize(
    objective,
    n_trials=TRIALS,
    n_jobs=JOBS,  # Utilize all available CPU cores
    callbacks=[custom_callback(TRIALS)]
)

# Print the best result
best_params = study.best_params
best_value = study.best_value
print("Best parameters found:")
for param, value in best_params.items():
    print(f"{param}: {value}")
print(f"--- Adjusted Value: {best_value:.2f}")
print(f"--- Sessions Reached: {study.best_trial.user_attrs['sessions']}")
print(f"--- Total Trades: {study.best_trial.user_attrs['trades']}")
print(f"--- Avg Margin: {study.best_trial.user_attrs['margin']:.2f}%")
print(f"--- Total Profit of: ${study.best_trial.user_attrs['profit']:,.2f}")

# Extract trial data manually
trials_data = []
for trial in study.trials:
    trial_dict = {
        'number': trial.number,'value': trial.value,
        'profit': trial.user_attrs.get('profit', np.nan),
        'trades': trial.user_attrs.get('trades', np.nan),
        'state': trial.state,'duration': trial.duration,
        'margin': trial.user_attrs.get('margin', np.nan),
        'sessions': trial.user_attrs.get('sessions', np.nan),
        'adjustment': trial.user_attrs.get('adjustment', np.nan),

    }

    # Include all parameters
    trial_dict.update(trial.params)
    trials_data.append(trial_dict)

# Create DataFrame
results_df = pd.DataFrame(trials_data)
# Reorder the DataFrame columns based on desired_order
desired_order = [
    'number','value','sessions','margin','profit','trades','b day','w day','prof','unprof','k_length','Slow_K_High','Slow_K_High_cap',
    'Slow_K_Low','Slow_K_Low_cap','candles','high_average','high_value','low_average','low_value','threshold', 
    'min_value','max_value','RoC_threshold','take_profit','stop_loss','timer','state','duration'
]

existing_order = [col for col in desired_order if col in results_df.columns]
results_df = results_df[existing_order]
results_df.to_parquet(os.path.join(output_dir, 'results.parquet'))

# Plot optimization history
fig1 = plot_optimization_history(study)
fig1.write_image(os.path.join(output_dir, 'optimization_history.png'), width=1600, height=900, scale=2)

# Plot parameter importances
fig2 = plot_param_importances(study)
fig2.write_image(os.path.join(output_dir, 'param_importances.png'), width=1600, height=900, scale=2)

fig1.update_layout(
    font=dict(size=16),  # Increase font size
    autosize=False,
    width=1920,
    height=1080,
)

fig1.write_html(os.path.join(output_dir, 'optimization_history.html'))
fig2.write_html(os.path.join(output_dir, 'param_importances.html'))

gc.collect()

# To view the HTML files, start a local HTTP server in the output directory
# cd /home/cheddarjackk/Developer/StrategyBuilder/output
# python3 -m http.server 8000

# Kill the server with the following commands:
# sudo lsof -i :8000
# sudo kill -9 <^PID # from above^>