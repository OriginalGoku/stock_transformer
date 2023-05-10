from pathlib import Path
import pandas as pd

CONFIG_SETTING = {
    'normalizer': 'z_normalize',
    'use_quantile_filter': True,
    'include_std': True,
    'use_mean_y': True,
    'window_size': 10,
    'forecast_size': 3,
    'ma_len': 5,
    'quantile_filter': 0.99,
    'source': 'LOW',
    'open_col_name': 'OPEN',
    'high_col_name': 'HIGH',
    'low_col_name': 'LOW',
    'close_col_name': 'CLOSE',
    'usable_data_col': ['close', 'open', 'high', 'low', 'Volume', 'Volume MA', 'RSI'],
    'data_folder': Path('data/daily'),
    'result_folder': Path('results'),
    'models_folder': Path('models'),
    'training_cut_off_date': pd.to_datetime('2018-01-03 09:30:00-05:00'),
    # 'file_list': ['BATS_M.csv'],
    'file_list': ['BATS_SPY.csv'],
    'threshold': 0.01
}