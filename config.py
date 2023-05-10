from dataclasses import dataclass, field
from typing import List

import pandas as pd
from pandas import DatetimeTZDtype
from pathlib import Path


@dataclass
class Config:
    normalizer: str = 'z_normalize'
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

    transformer_setting = {'epoc': 1, 'optimizer_choice': 'adamax', 'num_heads': 8, 'head_size': 256, 'ff_dim': 6,
                           'num_transformer_blocks': 6, 'mlp_units': 512, 'dropout': 0.5, 'mlp_dropout': 0.6,
                           'learning_rate': 0.00134, 'validation_split': 0.2, 'batch_size': 32}
    @property
    def file_name_format(self):
        return f"Window {self.window_size} - Forecast {self.forecast_size} - " \
               f"MA {self.ma_len} - Source {self.source} - {self.normalizer}"

    # Check if the folders exist, if not, create them after instantiating the class
    def __post_init__(self):
        for folder in [self.data_folder, self.result_folder, self.models_folder]:
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)

