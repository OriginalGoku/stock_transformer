from util import Util
from plots import Plotter
from config import Config
# from config_setting import CONFIG_SETTING
import pandas as pd


class ResultAnalyzer:
    def __init__(self, high_data, low_data, start_location_for_plotting, end_location_for_plotting, config: Config,
                 stock_file_name):
        print("Starting Result Analyzer")

        util = Util(config)

        # start_location_for_plotting = 100
        # end_location_for_plotting = 200

        stock = util.load_file(stock_file_name, config.data_folder)
        stock.index = pd.to_datetime(stock.index, utc=True)

        # could use either high or low data to get the start location
        print(f"low_data.iloc[0].name: {low_data.iloc[0].name}")
        print(f"low_data.iloc[0].date: {low_data.iloc[0].date}")
        print(f"low_data.iloc[0:3]: {low_data.iloc[0:3]}")
        print(f"low_data.iloc[0]: {low_data.iloc[0]}")
        start_date = low_data.iloc[0].date
        print(f"len(stock): {len(stock)}")
        print(f"start_date: {start_date}")
        print(f"stock.iloc[0:3] {stock.iloc[0:3]}")
        stock_for_plotting = stock.loc[start_date:]

        # file_name_format = f"Window {config.window_size} - Forecast {config.forecast_size} - MA {config.ma_len} - " \
        #                    f"Source {config.source} - {config.normalizer}"

        predict_df, result_dic, total_high_trades_above_threshold, total_low_trades_above_threshold = \
            util.analyze_high_low_trades(high_data, low_data, config.threshold)

        Plotter.plot_sample_predictions(stock_for_plotting, high_data, low_data, start_location_for_plotting,
                                        end_location_for_plotting, config.forecast_size, config.result_folder)

        Plotter.plot_trade_results(total_high_trades_above_threshold, total_low_trades_above_threshold,
                                   config.threshold, config.result_folder)

        Plotter.plot_explosive_moves(predict_df, config.result_folder)

        predict_df.to_csv(f"{config.result_folder}/Predictions {config.file_name_format}.csv")
        print(f"result_dic: {result_dic}")
        pd.DataFrame(result_dic, index=[0]).to_csv(f"{config.result_folder}/Results {config.file_name_format}.csv")
