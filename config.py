from dataclasses import dataclass, field
from typing import List

import pandas as pd
from pandas import DatetimeTZDtype
from pathlib import Path


@dataclass
class Config:
    normalizer: str = True
    use_quantile_filter: bool = False
    include_std: bool = True
    use_mean_y: bool = True
    window_size: int = 10
    forecast_size: int = 3
    ma_len: int = 5
    quantile_filter: float = 0.99

    source: str = 'CLOSE'
    open_col_name: str = 'OPEN'
    high_col_name: str = 'HIGH'
    low_col_name: str = 'LOW'
    close_col_name: str = 'CLOSE'
    usable_data_col: List[str] = field(default_factory=lambda: ['close', 'open', 'high', 'low', 'Volume', 'Volume MA', 'RSI'])

    data_folder: Path = Path('data/daily')
    result_folder: Path = Path('results')
    models_folder: Path = Path('models')

    training_cut_off_date: DatetimeTZDtype = pd.to_datetime('2018-01-03 09:30:00-05:00')
    file_list: List[str] = field(default_factory=lambda: ['BATS_SPY.csv'])

    # Trade Analyzer:
    threshold: float = 0.01

    file_name_format = f"Window {window_size} - Forecast {forecast_size} - MA {ma_len} - " \
                       f"Source {source} - {normalizer}"
