import pandas as pd
import numpy as np
# import parameters as param
from typing import List, Tuple
import keras
from pathlib import Path
from config import Config


# import random

def merge_data_dicts(data_dicts: List[dict]) -> dict:
    merged_data_dict = {key: [] for key in data_dicts[0].keys()}
    for data_dict in data_dicts:
        for key in data_dict.keys():
            # print(f"key = {key}, len(data_dict[key]) = {len(data_dict[key])}")
            merged_data_dict[key].extend(data_dict[key])
    return merged_data_dict

class Util:
    # def __init__(self, normalizer: str, file_list: List[str], window_size: int, train_cut_off_date: pd.DatetimeTZDtype,
    #              include_std: bool, source: str, file_name_format: str,
    #              open_col_name: str, high_col_name: str, low_col_name: str, close_col_name: str, result_folder: Path,
    #              data_folder: Path, usable_data_col: list, ma_len: int, use_mean_y: bool,
    #              forecast_size: int, use_quantile_filter: bool, quantile_filter: float):
    def __init__(self, config: Config):
        self.normalizer = config.normalizer
        self.file_list = config.file_list
        self.window_size = config.window_size
        self.train_cut_off_date = config.training_cut_off_date
        self.include_std = config.include_std
        self.source = config.source
        self.open_col_name = config.open_col_name
        self.high_col_name = config.high_col_name
        self.low_col_name = config.low_col_name
        self.close_col_name = config.close_col_name
        self.data_folder = config.data_folder
        self.usable_data_col = config.usable_data_col
        self.ma_len = config.ma_len
        self.use_mean_y = config.use_mean_y
        self.forecast_size = config.forecast_size
        self.use_quantile_filter = config.use_quantile_filter
        self.quantile_filter = config.quantile_filter
        self.file_name_format = config.file_name_format
        self.result_folder = config.result_folder

    # def convert_to_original(self, normalized_window, mean_data, std_data, z_normalize, x_0=None):
    #     if z_normalize:  # z_normalize = True
    #         print("normalized_window: ", len(normalized_window), "std_data: ", len(std_data), "mean_data: ", len(mean_data))
    #         original_window = [(normalized_window[i] * std_data[i]) + mean_data[i] for i in range(len(normalized_window))]
    #
    #     elif x_0 is not None:  # z_normalize = False
    #         original_window = [np.exp(normalized_window[i]) * x_0[i] for i in range(len(normalized_window))]
    #         # original_window = [(normalized_window[i] + 1) * x_0[i] for i in range(len(normalized_window))]
    #     else:
    #         return normalized_window
    #
    #     return np.array(original_window)
    @staticmethod
    def shuffle_reshape(X_train, X_test, y_train, y_test):
        idx = np.random.permutation(len(X_train))
        X_train = X_train[idx]
        y_train = y_train[idx]

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True)

        X_train = X_train.reshape(*X_train.shape, 1)
        X_test = X_test.reshape(*X_test.shape, 1)

        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")
        return X_train, X_test, y_train, y_test

    def convert_to_original(self, normalized_window, mean_data=None, std_data=None):

        if (self.normalizer == 'z_normalize') or (self.normalizer == 'z_normalize_pct_change'):
            print("normalized_window: ", len(normalized_window), "std_data: ", len(std_data), "mean_data: ",
                  len(mean_data))
            original_window = [(normalized_window[i] * std_data[i]) + mean_data[i] for i in
                               range(len(normalized_window))]

        elif self.normalizer == 'log':
            original_window = [np.exp(normalized_window[i]) for i in range(len(normalized_window))]
        # If no normalization is used, return the original window
        else:
            return normalized_window

        return np.array(original_window)

    def gen_multiple_sliding_window(self) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        train_data_dicts = []
        test_data_dicts = []

        for symbol_file in self.file_list:
            print(f"Processing {symbol_file}")
            data = self.load_file(symbol_file)
            data = self.fix_data(data)
            data_train = data[data.index < self.train_cut_off_date]
            data_test = data[data.index >= self.train_cut_off_date]

            print("Generating Train Data...")
            train_data_dict = self.gen_sliding_window_including_open(data_train)
            train_data_dict['symbol'] = [symbol_file.split('.')[0]] * len(train_data_dict['X'])
            train_data_dicts.append(train_data_dict)
            print("Generating Test Data...")
            test_data_dict = self.gen_sliding_window_including_open(data_test)
            test_data_dict['symbol'] = [symbol_file.split('.')[0].split('_')[1]] * len(test_data_dict['X'])
            test_data_dicts.append(test_data_dict)

        merged_train_data_dict = merge_data_dicts(train_data_dicts)
        merged_test_data_dict = merge_data_dicts(test_data_dicts)

        return tuple(np.array(values) for values in merged_train_data_dict.values()), \
            tuple(np.array(values) for values in merged_test_data_dict.values())



    def gen_sliding_window_including_open(self, data: pd.DataFrame) -> dict:
        # Description
        # This function generates a dictionary containing sliding windows of data from the input DataFrame for features
        # and target variables, normalizing them according to the specified method. It also extracts high, low, and
        # close values for each window, as well as their corresponding dates.
        # X has log of std if include_std is set to true as the first element, followed by data points and the last
        # element is the open of on the day of prediction of y.
        # the structure looks like this:
        # [np.log(std), stock['CLOSE'].iloc[:chunk_size], stock['OPEN'].iloc[chunk_size-1]]
        print(f"Generating Sliding Window on {self.source} - (window_size = {self.window_size}, normalizer = {self.normalizer})"
              f" len(data[{self.source}]) = {len(data)}")
        # Create a list containing column names for extraction

        columns = [self.source, self.open_col_name, self.high_col_name, self.low_col_name]
        # Generate sliding windows for features (X) and target (y) using stride_tricks

        sliding_window_x = np.lib.stride_tricks.sliding_window_view(data[self.source],
                                                                    window_shape=(self.window_size,))
        sliding_window_y = np.lib.stride_tricks.sliding_window_view(data[self.source].iloc[self.window_size:],
                                                                    window_shape=(self.forecast_size,))
        # Generate sliding windows for high, low, and close values

        high_window = np.lib.stride_tricks.sliding_window_view(data[self.high_col_name].iloc[self.window_size:],
                                                               window_shape=(self.forecast_size,))
        low_window = np.lib.stride_tricks.sliding_window_view(data[self.low_col_name].iloc[self.window_size:],
                                                              window_shape=(self.forecast_size,))
        close_window = np.lib.stride_tricks.sliding_window_view(data[self.close_col_name].iloc[self.window_size:],
                                                                window_shape=(self.forecast_size,))
        # Truncate data to match the sliding windows' shape
        # data = data[columns].iloc[self.window_size:]

        # Print lengths of sliding windows for debugging purposes
        print(f"len(sliding_window_x) = {len(sliding_window_x)}")
        print(f"len(sliding_window_y) = {len(sliding_window_y)}")

        # Calculate the number of windows to be generated
        n_windows = len(sliding_window_x) - self.forecast_size

        # Initialize a dictionary to store generated data
        data_dict = {'X': [], 'y': [], 'mean_x': [], 'std_x': [], 'mean_y': [], 'std_y': [], 'high': [], 'low': [],
                     'close': [], 'date': []}

        # Iterate through the range of windows to be generated
        for i in range(n_windows):

            # Calculate window_x by appending the open value to the current feature window
            window_x = np.append(sliding_window_x[i], data[self.open_col_name].iloc[i + self.window_size])

            # Normalize the window_x data based on the specified normalizer method
            if self.normalizer == 'z_normalize_pct_change':
                window_x = np.diff(window_x) / window_x[:-1]
                # window_y = np.diff(y_sliding_window[i]) / y_sliding_window[i][:-1]

            # Store mean and std of window_x to the data_dict
            if (self.normalizer == 'z_normalize') or (self.normalizer == 'z_normalize_pct_change'):
                mean_x = np.mean(window_x)
                std_x = np.std(window_x)
                # Store data to the data_dict
                data_dict['mean_x'].append(mean_x)
                data_dict['std_x'].append(std_x)
                normalized_window_x = (window_x - mean_x) / std_x

                if self.include_std:
                    # Apply natural logarithm to std to keep the scale of data in range
                    scaled_std_mean = np.log(std_x)
                    # Todo: Test to add mean_x to the beginning of the normalized_window_x
                    normalized_window_x = np.insert(normalized_window_x, 0, scaled_std_mean)

                # Normalize the target (y) data based on the specified normalizer method
                concat_window = np.concatenate((window_x[:-1], sliding_window_y[i]))
                mean_y = np.mean(concat_window)
                std_y = np.std(concat_window)
                # Store data to the data_dict
                data_dict['mean_y'].append(mean_y)
                data_dict['std_y'].append(std_y)

                if self.use_mean_y:
                    normalized_y = (np.mean(sliding_window_y[i]) - mean_y) / std_y
                else:
                    print("Z Normalized with single value of y")
                    normalized_y = (sliding_window_y[i][-1] - mean_y) / std_y
                    # if y is single value then use mean and std of X

            elif self.normalizer == 'log':
                normalized_window_x = np.log(window_x)
                if self.use_mean_y:
                    normalized_y = np.log(np.mean(sliding_window_y[i]))
                else:
                    normalized_y = np.log(sliding_window_y[i][-1])

            # No Normalization
            else:
                normalized_window_x = window_x

                if self.use_mean_y:
                    normalized_y = np.mean(sliding_window_y[i])
                else:
                    normalized_y = sliding_window_y[i][-1]

            data_dict['X'].append(normalized_window_x)
            data_dict['y'].append(normalized_y)
            data_dict['high'].append(np.max(high_window[i]))
            data_dict['low'].append(np.min(low_window[i]))
            data_dict['close'].append(close_window[i][-1])
            data_dict['date'].append(data.index[i])

        return data_dict

    # Improved Code
    def analyze_results(self, y_test, y_pred, X_test, date_test, high_test, low_test, close_test, save_results=True):
        # target is y_test
        df_analyze_results = pd.DataFrame({
            'date': date_test,
            'open': X_test[:, -1, 0].ravel(),
            'predict': y_pred.flatten(),
            'target': y_test.flatten(),
            'high': high_test.flatten(),
            'low': low_test.flatten(),
            'close': close_test.flatten()
        })

        df_analyze_results['open_to_prediction_difference'] = df_analyze_results['predict'] - df_analyze_results['open']
        df_analyze_results['log_open_to_prediction_difference'] = np.log(
            df_analyze_results['predict'] / df_analyze_results['open'])

        # Calculate the direction of the prediction
        cond1 = (df_analyze_results['target'] > df_analyze_results['open']) & (
                df_analyze_results['predict'] > df_analyze_results['open'])
        cond2 = (df_analyze_results['target'] < df_analyze_results['open']) & (
                df_analyze_results['predict'] < df_analyze_results['open'])
        df_analyze_results['direction'] = cond1 | cond2

        # Calculate 'log_trade_results' and 'original_trade_results'
        is_predict_gte_open = df_analyze_results['predict'] >= df_analyze_results['open']

        is_high_gte_predict = df_analyze_results['high'] >= df_analyze_results['predict']
        is_low_lte_predict = df_analyze_results['low'] <= df_analyze_results['predict']

        # Todo: Recalculate this for y_mean
        df_analyze_results['log_trade_results_open_to_close'] = np.where(is_predict_gte_open,
                                                                         np.log(
                                                                             df_analyze_results['target'] /
                                                                             df_analyze_results['open']),
                                                                         np.log(
                                                                             df_analyze_results['open'] /
                                                                             df_analyze_results['target']))

        # Calculate the difference between the prediction and the last value
        df_analyze_results['log_diff_predict_original'] = np.log(
            df_analyze_results['predict'] / df_analyze_results['target'])

        # Calculate cumulative trade results
        df_analyze_results['cum_log_trade_results_open_to_close'] = df_analyze_results[
            'log_trade_results_open_to_close'].cumsum()
        # df_analyze_results['cum_original_trade_results'] = df_analyze_results['original_trade_results'].cumsum()

        # Check high and low prediction
        df_analyze_results['correct_prediction'] = (
                (is_predict_gte_open & (high_test >= df_analyze_results['predict'])) |
                (~is_predict_gte_open & (low_test <= df_analyze_results['predict']))
        )

        # Print the number of wrong direction trades
        print(
            f"Wrong Direction trades: {(~df_analyze_results['correct_prediction']).sum()} out of {len(df_analyze_results)}"
            f" [ {round(100 * ((~df_analyze_results['correct_prediction']).sum() / len(df_analyze_results)), 2)}% ]")

        # Update the trade_result_including_high_low column based on the correct_prediction column
        df_analyze_results = df_analyze_results.assign(
            trade_result_including_high_low=[
                row['log_trade_results_open_to_close'] if not row['correct_prediction']
                else np.abs(row['predict'] - row['open']) / row['open']
                for _, row in df_analyze_results.iterrows()
            ]
        )

        # Calculate the cumulative trade result including high and low
        df_analyze_results['cum_trade_result_including_high_low'] = \
            df_analyze_results['trade_result_including_high_low'].cumsum()

        # Save results to a CSV file
        file_name = Path('Results ' + self.file_name_format + '.csv')
        save_path = self.result_folder / file_name
        if save_results:
            self.save_csv(df_analyze_results, save_path)

        return df_analyze_results

    @staticmethod
    def analyze_high_low_trades(high_data, low_data, threshold: float):
        # Description: The function processes the data to analyze and summarize trading performance based on the
        # provided threshold.
        #
        # The function first creates a new DataFrame called predict_df by concatenating required columns from the input
        # DataFrames. It calculates log_open_to_low and log_open_to_high columns for the predict_df. It computes
        # result_short_using_low and result_long_using_high columns using the apply method and a lambda function. The
        # function then checks for trades above the threshold, and calculates the number of short and long trades above
        # the threshold and their respective loss rates. It calculates the number of unique trading days above the
        # threshold. The function identifies the top 20 worst and best short and long trades. It calculates the total
        # return from trades above the threshold, as well as the total return from explosive long and short trades.
        # Finally, the function returns the predict_df and the calculated total returns for short, long, explosive long,
        # and explosive short trades.
        rounding_precision = 3

        predict_df = pd.concat(
            [
                high_data[['open', 'high', 'low', 'close', 'predict']].rename(columns={'predict': 'predict_high'}),
                low_data['predict'].rename('predict_low')
            ],
            axis=1
        )

        # Calculate log_open_to_low and log_open_to_high
        predict_df['log_open_to_low'] = np.log(predict_df['predict_low'] / predict_df['open'])
        predict_df['log_open_to_high'] = np.log(predict_df['predict_high'] / predict_df['open'])

        # Compute result_short_using_low and result_long_using_high using the apply method and a lambda function
        predict_df['result_short_using_low'] = predict_df.apply(
            lambda row: np.abs(row['log_open_to_low']) if row['low'] <= row['predict_low']
            else np.log(row['open'] / row['close']),
            axis=1
        )

        predict_df['result_long_using_high'] = predict_df.apply(
            lambda row: np.abs(row['log_open_to_high']) if row['high'] >= row['predict_high']
            else np.log(row['close'] / row['open']),
            axis=1
        )

        # Check trades Above threshold
        total_low_trades_above_threshold = predict_df.query('log_open_to_low < -@threshold')
        total_high_trades_above_threshold = predict_df.query('log_open_to_high > @threshold')

        short_losing_mask = total_low_trades_above_threshold['result_short_using_low'] < 0
        short_trade_loss = total_low_trades_above_threshold[short_losing_mask]
        short_lost_rate = round(len(short_trade_loss) / len(total_low_trades_above_threshold), rounding_precision)

        long_losing_mask = total_high_trades_above_threshold['result_long_using_high'] < 0
        long_trade_loss = total_high_trades_above_threshold[long_losing_mask]
        long_lost_rate = round(len(long_trade_loss) / len(total_high_trades_above_threshold), rounding_precision)

        print(
            f"Total number of Short trades for Threshold < -{100 * threshold}% is "
            f"{len(total_low_trades_above_threshold)} "
            f"[ {round(100 * len(total_low_trades_above_threshold) / len(predict_df), 2)}% ] "
            f"-> Loss {short_lost_rate}% (No of Losing Trades: {len(short_trade_loss)})")
        print(
            f"Total number of Long trades for Threshold > {100 * threshold}% is "
            f"{len(total_high_trades_above_threshold)} "
            f"[ {round(100 * len(total_high_trades_above_threshold) / len(predict_df), 2)}% ] "
            f"-> Loss {long_lost_rate}% (No of Losing Trades: {len(long_trade_loss)})")

        # Calculate number of unique trades above threshold
        combined_dates = pd.concat([pd.Series(total_low_trades_above_threshold.index),
                                    pd.Series(total_high_trades_above_threshold.index)])
        unique_trading_days = combined_dates.nunique()
        print(f"Unique Trading Days above threshold: {unique_trading_days}")

        worst_20_short = round(
            total_low_trades_above_threshold[short_losing_mask].sort_values(by='result_short_using_low').head(20)[
                'result_short_using_low'].sum(), rounding_precision)
        worst_20_long = round(
            total_high_trades_above_threshold[long_losing_mask].sort_values(by='result_long_using_high').head(20)[
                'result_long_using_high'].sum(), rounding_precision)

        best_20_short = round(
            total_low_trades_above_threshold[~short_losing_mask].sort_values(by='result_short_using_low',
                                                                             ascending=False).head(20)[
                'result_short_using_low'].sum(), rounding_precision)
        best_20_long = round(
            total_high_trades_above_threshold[~long_losing_mask].sort_values(by='result_long_using_high',
                                                                             ascending=False).head(20)[
                'result_long_using_high'].sum(), rounding_precision)

        print(f"Sum of Worst 20 Short Trades: {worst_20_short}%")
        print(f"Sum of Worst 20 Long Trades: {worst_20_long}%")
        print(f"Sum of Best 20 Short Trades: {best_20_short}%")
        print(f"Sum of Best 20 Long Trades: {best_20_long}%")

        # Total Return from trades above threshold
        total_return_short_using_low = total_low_trades_above_threshold.result_short_using_low.cumsum().to_list()[-1]
        total_return_long_using_high = total_high_trades_above_threshold.result_long_using_high.cumsum().to_list()[-1]

        # Explosive move predictions
        # Low higher than open:
        prediction_low_higher_than_open = predict_df[predict_df['log_open_to_low'] > 0]['result_long_using_high']
        total_return_explosive_long = prediction_low_higher_than_open.cumsum().to_list()[-1]

        prediction_high_lower_than_open = predict_df[predict_df['log_open_to_high'] < 0]['result_short_using_low']
        total_return_explosive_short = prediction_high_lower_than_open.cumsum().to_list()[-1]

        print(
            f"Total Return from {len(total_low_trades_above_threshold)} Short Trades: "
            f"{round(100 * total_return_short_using_low, 2)}% "
            f"(average: {round(100 * total_return_short_using_low / len(total_low_trades_above_threshold), 2)}%)")
        print(
            f"Total Return from {len(total_high_trades_above_threshold)} Long Trades: "
            f"{round(100 * total_return_long_using_high, 2)}% "
            f"(average: {round(100 * total_return_long_using_high / len(total_high_trades_above_threshold), 2)}%)")
        print(
            f"Total Return from {len(prediction_low_higher_than_open)} Explosive Long Trades: "
            f"{round(100 * total_return_explosive_long, 2)}% "
            f"(average: {round(100 * total_return_explosive_long / len(prediction_low_higher_than_open), 2)}%)")
        print(
            f"Total Return from {len(prediction_high_lower_than_open)} "
            f"Explosive Short Trades: {round(100 * total_return_explosive_short, 2)}% "
            f"(average: {round(100 * total_return_explosive_short / len(prediction_high_lower_than_open), 2)}%)")

        # Initialize the results dictionary
        results = {
            'total_low_trades_above_threshold': len(total_low_trades_above_threshold),
            'total_high_trades_above_threshold': len(total_high_trades_above_threshold),
            'short_lost_rate': short_lost_rate,
            'long_lost_rate': long_lost_rate,
            'unique_trading_days': unique_trading_days,
            'worst_20_short': worst_20_short,
            'worst_20_long': worst_20_long,
            'best_20_short': best_20_short,
            'best_20_long': best_20_long,
            'total_return_short_using_low': round(total_return_short_using_low, rounding_precision),
            'total_return_long_using_high': round(total_return_long_using_high, rounding_precision),
            'explosive_long_no_trades': len(prediction_low_higher_than_open),
            'total_return_explosive_long': round(total_return_explosive_long, rounding_precision),
            'explosive_short_no_trades': len(prediction_high_lower_than_open),
            'total_return_explosive_short': round(total_return_explosive_short, rounding_precision),
            'avg_return_short': round(total_return_short_using_low / len(total_low_trades_above_threshold), rounding_precision),
            'avg_return_long': round(total_return_long_using_high / len(total_high_trades_above_threshold), rounding_precision),
            'avg_return_explosive_long': round(total_return_explosive_long / len(prediction_low_higher_than_open), rounding_precision),
            'avg_return_explosive_short': round(total_return_explosive_short / len(prediction_high_lower_than_open), rounding_precision)
        }

        # Return the results dictionary along with the other return values
        return predict_df, results, total_high_trades_above_threshold, total_low_trades_above_threshold

    def load_file(self, file_name: str, folder_path: Path = None):
        if folder_path is None:
            file_path = self.data_folder / file_name
        else:
            file_path = folder_path / file_name

        print(f"Loading file: {file_path}")
        data = pd.read_csv(file_path, index_col=[0], parse_dates=True).rename_axis('Date')
        return data

    def fix_data(self, data):
        # Todo: We can use .loc to fix the data instead of making a copy
        data = data.copy()
        data = data[self.usable_data_col]
        data.columns = [x.upper() for x in data.columns]
        data['CLOSE_MA'] = data['CLOSE'].rolling(self.ma_len).mean()
        data['PCT_CHANGE'] = data['CLOSE'].pct_change()
        data['HLCC'] = data['HIGH'] + data['LOW'] + data['CLOSE'] + data['CLOSE']
        data['OHLC'] = data['OPEN'] + data['HIGH'] + data['LOW'] + data['CLOSE']
        data['DETREND'] = data['CLOSE'] - data['CLOSE_MA']
        data['LOG_DETREND'] = np.log(data['CLOSE']) - np.log(data['CLOSE_MA'])
        if self.use_quantile_filter:
            data['LOG_DETREND'] = np.clip(data['LOG_DETREND'], data['LOG_DETREND'].quantile(1 - self.quantile_filter),
                                          data['LOG_DETREND'].quantile(self.quantile_filter))
        data.dropna(inplace=True)

        print(f"Loaded {len(data)} rows")
        print(f"Columns: {data.columns}")
        return data

    @staticmethod
    def save_csv(data, file_name):
        data.to_csv(file_name, index=None)

    def load_high_low_results(self, high_results_file_name, low_results_file_name, folder_path):
        print("load_high_low_results function Not implemented yet")
        low_data = self.load_file(low_results_file_name, folder_path)
        high_data = self.load_file(high_results_file_name, folder_path)

        return high_data, low_data
