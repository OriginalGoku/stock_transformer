import matplotlib.pyplot as plt
# import parameters as param
import numpy as np
import keras
import pandas as pd
from pathlib import Path
import mplfinance as mpf


class Plotter:
    def __init__(self, y_test, y_pred, result_folder: Path, history: keras.callbacks.History, file_name_format: str,
                 save_results=True, display_plots=True):
        self.result_folder = result_folder
        self.save_results = save_results
        self.plot_file_details = file_name_format
        self.history = history
        self.y_test = y_test
        self.y_pred = y_pred
        self.display_plots = display_plots

    @staticmethod
    def plot_hist_distribution(y_train, y_test, first_input_title, second_input_title, result_folder, bins=100):
        # Plot the histogram for the train and test set y values

        hist_train, bins_train = np.histogram(y_train, bins=bins)
        hist_test, bins_test = np.histogram(y_test, bins=bins)

        plt.bar(bins_train[:-1], hist_train, width=(bins_train[1] - bins_train[0]), label=first_input_title)
        plt.bar(bins_test[:-1], hist_test, width=(bins_test[1] - bins_test[0]), label=second_input_title)
        plt.legend()
        file_name = "Train and Test y Distribution.png"
        plt.savefig(result_folder / file_name)
        plt.show()

    def plot_train_validation_loss(self):
        # Plot the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Loss Improvement of the Network')
        plt.legend()
        if self.save_results:
            plt.savefig(Path(self.result_folder / 'Network train and validation loss.png'))
        plt.show()

    def plot_history_metrics(self):
        total_plots = len(self.history.history)
        cols = total_plots // 2

        rows = total_plots // cols

        if total_plots % cols != 0:
            rows += 1

        pos = range(1, total_plots + 1)
        plt.figure(figsize=(15, 10))
        for i, (key, value) in enumerate(self.history.history.items()):
            plt.subplot(rows, cols, pos[i])
            plt.plot(range(len(value)), value)
            plt.title(str(key))
        if self.save_results:
            plt.savefig(Path(self.result_folder / 'Network history metrics.png'))
        plt.show()

    def plot_scatter_true_vs_predicted(self, start_: int, end_: int):
        fig = plt.figure(figsize=(30, 10))
        print(f"Plotting from {start_} to {end_} for y_test = {len(self.y_test)}, predictions = {len(self.y_pred)} ")
        # Plot the limited range of true values vs the predicted values
        plt.scatter(np.arange(start_, end_), self.y_pred[start_:end_], alpha=0.5, marker='x', color='red',
                    label='Predicted')
        plt.scatter(np.arange(start_, end_), self.y_test.reshape(-1, 1)[start_:end_], alpha=0.5, marker='o',
                    color='blue',
                    label='True')
        plt.ylabel("Predicted/True Values")
        plt.title("True Values vs Predicted Values")
        plt.legend()
        file_name = Path('Scatter True vs Predict' + self.plot_file_details + ".png")

        if self.save_results:
            plt.savefig(self.result_folder / file_name)
        plt.show()
        # if self.display_plots:
        # else:
        #     plt.close()

    def plot_histogram_y_test_minus_y_pred(self, bins=30):
        # Calculate the differences between true and predicted values
        differences = (self.y_test - self.y_pred.reshape(-1, )).flatten()
        # differences_pct = np.round([differences[i] / x_test[i][-1] for i in range(len(x_test))], 4).flatten()
        differences_pct = np.round([np.log(self.y_pred.reshape(-1, )[i] / self.y_test[i])
                                    for i in range(len(self.y_test))], 4).flatten()
        # Plot the histogram of differences
        differences_pct = np.clip(differences_pct, -1, 1)
        plt.hist(differences_pct, bins=bins, color='purple')
        plt.xlabel("Difference")
        plt.ylabel("Frequency")
        plt.title("Histogram of Differences between True and Predicted Values")
        file_name = 'Histogram True-Predict' + self.plot_file_details + ".png"
        if self.save_results:
            plt.savefig(self.result_folder / file_name)
        plt.show()
        # if self.display_plots:
        # else:
        #     plt.close()

    def plot_scatter_true_vs_predicted_diagonal(self):
        # Plot the true values vs the predicted values
        plt.scatter(self.y_test, self.y_pred, alpha=0.5)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        # f"True Values vs Predicted Values\nChunk Len:{param.chunk_size} - SMA{param.ma_len} - Future Len:{param.forecast_size}")
        plt.title(f"True Values vs Predicted Values \n{self.plot_file_details}")
        plt.plot([min(self.y_test), max(self.y_test)], [min(self.y_test), max(self.y_test)],
                 color='red')  # Diagonal line
        file_name = 'Scatter - True vs Predicted' + self.plot_file_details + ".png"
        if self.save_results:
            plt.savefig(
                Path(self.result_folder / file_name))
        plt.show()
        # if self.display_plots:
        # else:
        #     plt.close()

    # def plot_scatter_true_vs_predicted_diagonal_only_different_sign(y_test, y_pred, save_results=True):
    #     # Plot the true values vs the predicted values
    #     result_indices = [index for index, (test_val, pred_val) in enumerate(zip(y_test, y_pred)) if
    #                       np.sign(test_val) != np.sign(pred_val)]
    #
    #     if result_indices:  # Check if result_indices is not empty
    #         plt.scatter(y_test[result_indices], y_pred[result_indices], alpha=0.5)
    #         plt.xlabel("True Values")
    #         plt.ylabel("Predicted Values")
    #         plt.title(
    #             f"True Values vs Predicted (Wrong Direction only)\nChunk Len:{param.chunk_size} - SMA{param.ma_len} - Future "
    #             f"Len:{param.forecast_size}")
    #         # Diagonal line
    #         plt.plot([min(y_test[result_indices]), max(y_test[result_indices])], [min(y_test[result_indices]),
    #                                                                               max(y_test[result_indices])], color='red')
    #         if save_results:
    #             file_name = 'True vs Predicted (Wrong Direction)' + param.plot_file_details
    #             print("Saving file: ", file_name)
    #             # print("Saving full path: " + param.result_folder + "/" + file_name)
    #             plt.savefig(param.result_folder + "/" + file_name)
    #         plt.show()
    #     else:
    #         print("No data points with different signs between true and predicted values.")
    #

    def plot_cum_log_return(self, results: pd.DataFrame):
        results['cum_log_trade_results_open_to_close'].plot(label='Only Close')
        results['cum_trade_result_including_high_low'].plot(label='Including High/Low')
        plt.legend()
        file_name = "Cumulative Log Return " + self.plot_file_details + ".png"
        plt.savefig(self.result_folder / file_name)

    @staticmethod
    def generate_plot_data(low_data, high_data, start_location_for_plotting, end_location_for_plotting, forecast_size):
        len_prediction_lines = len(
            low_data.iloc[start_location_for_plotting:end_location_for_plotting].index) - forecast_size + 1
        data_temp_low = low_data.iloc[start_location_for_plotting:end_location_for_plotting]
        data_temp_high = high_data.iloc[start_location_for_plotting:end_location_for_plotting]

        date_range_low = [
            [(data_temp_low.index[i], data_temp_low['predict'][i]),
             (data_temp_low.index[i + forecast_size - 1], data_temp_low['predict'][i])]
            for i in range(len_prediction_lines)
        ]
        color_low = ['r'] * len_prediction_lines

        date_range_high = [
            [(data_temp_high.index[i], data_temp_high['predict'][i]),
             (data_temp_high.index[i + forecast_size - 1], data_temp_high['predict'][i])]
            for i in range(len_prediction_lines)
        ]
        color_high = ['g'] * len_prediction_lines

        full_data = date_range_low + date_range_high
        full_color = color_low + color_high
        return full_data, full_color

    @staticmethod
    def plot_sample_predictions(stock_for_plotting, high_data, low_data, start_location_for_plotting,
                                end_location_for_plotting, forecast_size, result_folder):
        start = start_location_for_plotting
        end = end_location_for_plotting
        results, colors = Plotter.generate_plot_data(low_data, high_data, start, end, forecast_size)

        print(results)
        print(colors)


        print(f"len(results): {len(results)}")
        print(f"len(colors): {len(colors)}")
        print(f"len(stock_for_plotting): {len(stock_for_plotting)}")
        print(f"start: {start}")
        print(f"end: {end}")
        print(f"stock_for_plotting.iloc[start:end]: {stock_for_plotting.iloc[start:end]}")

        file_name = result_folder / "Sample Prediction.png"
        # TODO: fix this
        # mpf.plot(stock_for_plotting.iloc[start:end], savefig=file_name,
        #          type='candle', figsize=(20, 10), style='yahoo',
        #          alines=dict(alines=results, colors=colors, linewidths=2, alpha=0.2))

    @staticmethod
    def plot_trade_results(total_high_trades_above_threshold, total_low_trades_above_threshold, threshold,
                           result_folder):
        plt.plot(total_low_trades_above_threshold.result_short_using_low.cumsum().to_list(), label="Short")
        plt.plot(total_high_trades_above_threshold.result_long_using_high.cumsum().to_list(), label="Long")
        plt.ylabel("% Cum Return")
        plt.xlabel("Trades")
        plt.title(f"Cummulative Profit for trades greater than +-{100 * threshold}%")
        plt.legend()
        file_name = result_folder / "Trade Results.png"
        plt.savefig(file_name)
        plt.show()

    import matplotlib.pyplot as plt
    @staticmethod
    def plot_explosive_moves(predict_df, result_folder):
        positive_prediction = predict_df[predict_df['log_open_to_low'] > 0]['result_long_using_high']
        negative_prediction = predict_df[predict_df['log_open_to_high'] < 0]['result_short_using_low']

        fig, axes = plt.subplots(2, 1)#, figsize=(10, 10))

        # Very positive prediction plot
        axes[0].set_ylabel("% Cumsum")
        axes[0].set_title(
            f"Very Positive Prediction (Predicted Low>Open)\n[No of Trades: {len(positive_prediction)}]")
        axes[0].plot(positive_prediction.cumsum(), label='Long')
        axes[0].legend()

        # Very negative prediction plot
        axes[1].set_ylabel("% Cumsum")
        axes[1].set_xlabel("Trade Date")
        axes[1].set_title(
            f"Very Negative Prediction (Predicted High<Open)\n[No of Trades: {len(negative_prediction)}]")
        axes[1].plot(negative_prediction.cumsum(), label='Short', color='orange')

        axes[1].legend()

        plt.tight_layout()
        file_name = result_folder / "Explosive Moves.png"
        plt.savefig(file_name)
        plt.show()
