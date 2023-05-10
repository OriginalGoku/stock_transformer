import transformer
# import optuna
# from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
from plots import Plotter
from util import Util
from config_data import Config
from config_setting import CONFIG_SETTING
from result_analyzer import ResultAnalyzer

TRANSFORMER_SETTING = {'epoc': 1, 'optimizer_choice': 'adamax', 'num_heads': 8, 'head_size': 256, 'ff_dim': 6,
                       'num_transformer_blocks': 6, 'mlp_units': 512, 'dropout': 0.5, 'mlp_dropout': 0.6,
                       'learning_rate': 0.00134, 'validation_split': 0.2, 'batch_size': 32}


def generate_data(util: Util, config: Config):
    train_data, test_data = util.gen_multiple_sliding_window()

    X_train, y_train, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, low_train, close_train, \
        date_train, symbol_train = train_data

    X_test, y_test, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, low_test, close_test, date_test, \
        symbol_test = test_data

    X_train, X_test, y_train, y_test = util.shuffle_reshape(X_train, X_test, y_train, y_test)

    Plotter.plot_hist_y_distribution(y_train, y_test, config.result_folder)

    return X_train, X_test, y_train, y_test, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, \
        low_train, close_train, date_train, symbol_train, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, \
        low_test, close_test, date_test, symbol_test


def test_model(model_high_name, model_low_name, util, config: Config, start_location_for_plotting,
               end_location_for_plotting,
               stock_file_name, file_name_format):
    print("Loading models...")
    model_high = transformer.load_model(Path(config.models_folder / model_high_name))
    model_low = transformer.load_model(Path(config.models_folder / model_low_name))

    config.source = 'HIGH'

    X_train, X_test, y_train, y_test, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, \
        low_train, close_train, date_train, symbol_train, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, \
        low_test, close_test, date_test, symbol_test = generate_data(util, config)

    print("Generating results from high model...")

    y_pred_normal = model_high.predict(X_test)
    y_pred = util.convert_to_original(y_pred_normal, mean_y_test, std_y_test)
    y_test = util.convert_to_original(y_test, mean_y_test, std_y_test)
    X_test = util.convert_to_original(X_test, mean_x_test, std_x_test)

    plotter = Plotter(y_test, y_pred, config.result_folder, model_high, file_name_format)

    plotter.plot_scatter_true_vs_predicted(0, len(y_test) // 3)
    plotter.plot_histogram_y_test_minus_y_pred()
    plotter.plot_scatter_true_vs_predicted_diagonal()

    high_data = util.analyze_results(y_test, y_pred, X_test, date_test, high_test, low_test, close_test)

    print("Generating results from low model...")

    config.source = 'LOW'

    X_train, X_test, y_train, y_test, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, \
        low_train, close_train, date_train, symbol_train, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, \
        low_test, close_test, date_test, symbol_test = generate_data(util, config)

    y_pred_normal = model_low.predict(X_test)
    y_pred = util.convert_to_original(y_pred_normal, mean_y_test, std_y_test)
    y_test = util.convert_to_original(y_test, mean_y_test, std_y_test)
    X_test = util.convert_to_original(X_test, mean_x_test, std_x_test)

    plotter = Plotter(y_test, y_pred, config.result_folder, model_low, file_name_format)

    plotter.plot_scatter_true_vs_predicted(0, len(y_test) // 3)
    plotter.plot_histogram_y_test_minus_y_pred()
    plotter.plot_scatter_true_vs_predicted_diagonal()

    low_data = util.analyze_results(y_test, y_pred, X_test, date_test, high_test, low_test, close_test)

    result_analyzer = ResultAnalyzer(high_data, low_data, start_location_for_plotting, end_location_for_plotting,
                                     config, stock_file_name, file_name_format)


def train_model(X_train, y_train, config):
    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, window_size=config.window_size,
                                                       forecast_size=config.forecast_size, source=config.source,
                                                       data_folder=config.data_folder, normalizer=config.normalizer,
                                                       model_folder=config.models_folder, **TRANSFORMER_SETTING)
    # transformer.evaluate_model(model, X_test, y_test)
    return model


def train(util: Util, config):
    X_train, X_test, y_train, y_test, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, \
        low_train, close_train, date_train, symbol_train, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, \
        low_test, close_test, date_test, symbol_test = generate_data(util, config)

    model = train_model(X_train, y_train, config)

    y_pred_normal = model.predict(X_test)
    print(f"converting y_pred_normal to original")
    y_pred = util.convert_to_original(y_pred_normal, mean_y_test, std_y_test)
    print(f"converting y_test to original")
    y_test = util.convert_to_original(y_test, mean_y_test, std_y_test)
    print(f"converting X_test to original")
    X_test = util.convert_to_original(X_test, mean_x_test, std_x_test)

    plotter = Plotter(y_test, y_pred, config.result_folder, model, file_name_format)

    # Make sure to check the len of y_test
    plotter.plot_scatter_true_vs_predicted(0, len(y_test) // 3)
    plotter.plot_histogram_y_test_minus_y_pred()
    plotter.plot_scatter_true_vs_predicted_diagonal()
    # plotter.plot_scatter_true_vs_predicted_diagonal_only_different_sign()
    temp_res = pd.DataFrame()
    temp_res['date'] = date_test
    temp_res['symbol'] = symbol_test
    temp_res['open'] = [X_test[i][-1][0] for i in range(len(X_test))]
    temp_res['y_test'] = y_test
    temp_res['y_pred'] = y_pred
    temp_res['high'] = high_test
    temp_res['low'] = low_test
    temp_res['close'] = close_test
    temp_res['open_original'] = [(X_test[i][-1][0] * std_x_test[i]) + mean_x_test[i] for i in range(len(X_test))]
    temp_res['mean_x'] = mean_x_test
    temp_res['std_x'] = std_x_test
    temp_res['mean_y'] = mean_y_test
    temp_res['std_y'] = std_y_test
    temp_res['y_test_original'] = (y_test * std_y_test) + mean_y_test
    temp_res['y_pred_original'] = ((y_pred.flatten()) * std_y_test) + mean_y_test

    file_name = file_name_format + "Raw Results.csv"
    temp_res.to_csv(config.result_folder / file_name, index=False)
    results = util.analyze_results(y_test, y_pred, X_test, date_test, high_test, low_test, close_test)
    plotter.plot_cum_log_return(results)


# Call the main function
if __name__ == "__main__":
    main_config = Config(**CONFIG_SETTING)
    model_high_name = "daily Window size [10] Forecast [3] Source [HIGH] z_normalize.h5"
    model_low_name = "daily Window size [10] Forecast [3] Source [LOW] z_normalize.h5"
    start_location_for_plotting = 0
    end_location_for_plotting = 10
    # stock_file_name = "BATS_SPY.csv"
    training_cut_off_date = pd.to_datetime('2011-01-03 09:30:00-05:00')
    main_config.file_list = ["BATS_TSLA.csv"]

    file_name_format = f"Window {main_config.window_size} - Forecast {main_config.forecast_size} - MA {main_config.ma_len} - " \
                       f"Source {main_config.source} - {main_config.normalizer}"
    main_config.training_cut_off_date = training_cut_off_date

    utility = Util(main_config.normalizer, main_config.file_list, main_config.window_size,
                   main_config.training_cut_off_date,
                   main_config.include_std, main_config.source, file_name_format, main_config.open_col_name,
                   main_config.high_col_name,
                   main_config.low_col_name, main_config.close_col_name, main_config.result_folder,
                   main_config.data_folder,
                   main_config.usable_data_col, main_config.ma_len, main_config.use_mean_y, main_config.forecast_size,
                   main_config.use_quantile_filter, main_config.quantile_filter)

    # main('train', util, config, file_name_format)
    #Todo: Check by number of long trades are so little
    test_model(model_high_name, model_low_name, utility, main_config, start_location_for_plotting,
               end_location_for_plotting,
               main_config.file_list[0], file_name_format)

# Testing Git