from dataclasses import dataclass
from typing import List
from pandas import DatetimeTZDtype
from pathlib import Path


@dataclass
class Config:
    normalizer: str
    use_quantile_filter: bool
    include_std: bool
    use_mean_y: bool
    window_size: int
    forecast_size: int
    ma_len: int
    quantile_filter: float

    source: str
    open_col_name: str
    high_col_name: str
    low_col_name: str
    close_col_name: str
    usable_data_col: List[str]

    data_folder: Path
    result_folder: Path
    models_folder: Path

    training_cut_off_date: DatetimeTZDtype
    file_list: List[str]

    # Trade Analyzer:
    threshold: float
