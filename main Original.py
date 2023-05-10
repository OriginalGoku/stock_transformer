import transformer
import plots
import optuna
# from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from pathlib import Path
from plots import Plotter
from util import Util
import configparser

# ORIGINAL_TRANSFORMER_SETTING = {"epoc": 1,
#                                 "num_heads": 1,#4,
#                                 "head_size": 256,
#                                 "ff_dim": 4,
#                                 "num_transformer_blocks": 4,
#                                 "mlp_units": 128,
#                                 "dropout": 0.4,
#                                 "mlp_dropout": 0.25,
#                                 "optimizer_choice": 'adam',
#                                 "loss": 'mean_squared_error',
#                                 "metrics": 'mean_absolute_error',
#                                 "learning_rate": 0.001,
#                                 "min_learning_rate": 0.00001,
#                                 "print_summary": True,
#                                 "validation_split": 0.2,
#                                 "batch_size": 32}
# TRANSFORMER_SETTING = {'epoc': 4, 'optimizer_choice': 'adam', 'num_heads': 4, 'head_size': 256, 'ff_dim': 3,
#                        'num_transformer_blocks': 3, 'mlp_units': 512, 'dropout': 0.2, 'mlp_dropout': 0.5,
#                        'learning_rate': 0.00092, 'validation_split': 0.5, 'batch_size': 32}

# Optimized using Optuna 3 May 2023

# For NOT Z Normalized (150 Trials)
# TRANSFORMER_SETTING = {'optimizer_choice': 'nadam', 'num_heads': 5, 'head_size': 512, 'ff_dim': 5, 'num_transformer_blocks': 2,
#        'mlp_units': 512, 'dropout': 0.3, 'mlp_dropout': 0.1, 'learning_rate': 0.00735,
#        'validation_split': 0.3, 'batch_size': 64}

# for Z Normalized (50 Trials)
# TRANSFORMER_SETTING = {'epoc': 1, 'optimizer_choice': 'adamax', 'num_heads': 3, 'head_size': 256, 'ff_dim': 3,
#                        'num_transformer_blocks': 4, 'mlp_units': 512, 'dropout': 0.2, 'mlp_dropout': 0.6,
#                        'learning_rate': 0.00134, 'validation_split': 0.1, 'batch_size': 32}

TRANSFORMER_SETTING = {'epoc': 1, 'optimizer_choice': 'adamax', 'num_heads': 8, 'head_size': 256, 'ff_dim': 6,
                       'num_transformer_blocks': 6, 'mlp_units': 512, 'dropout': 0.5, 'mlp_dropout': 0.6,
                       'learning_rate': 0.00134, 'validation_split': 0.2, 'batch_size': 32}


# training_cut_off_date = pd.to_datetime('2019-01-03 09:30:00-05:00')
#
# X, Y, X_TEST, Y_TEST, train_mean, train_std, test_mean, test_std, x_0_train, x_0_test = util.gen_multiple_sliding_window(
#         param.files, param.chunk_size,
#         param.z_normalize,
#         training_cut_off_date, 'CLOSE')

def objective(trial):
    idx = np.random.permutation(len(X))
    X_train = X[idx]
    y_train = Y[idx]

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_TEST.reshape(X_TEST.shape[0], X_TEST.shape[1], 1)
    # data = util.load_file('data/BATS_SPY.csv')
    # X, y = util.gen_sliding_window(data, param.chunk_size, param.z_normalize)
    # X_train, X_test, y_train, y_test = util.generate_random_sets(util.load_file('data/BATS_SPY.csv'), len_test=300,
    #                                                              test_pct=0.3)

    optimizer = trial.suggest_categorical("optimizer_choice",
                                          ['sgd', 'adam', 'rmsprop', 'adagrad', 'adadelta', 'adamax', 'nadam', 'ftrl'])
    num_head = trial.suggest_int("num_heads", 1, 5)
    head_size = trial.suggest_categorical("head_size", [128, 256, 512])
    ff_dim = trial.suggest_int("ff_dim", 1, 5)
    num_transformer_blocks = trial.suggest_int("num_transformer_blocks", 1, 5)
    mlp_units = trial.suggest_categorical("mlp_units", [128, 256, 512])
    dropout = trial.suggest_float("dropout", 0.1, 0.6, step=0.1)
    mlp_dropout = trial.suggest_float("mlp_dropout", 0.1, 0.6, step=0.1)
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.01, step=0.00001)
    validation_split = trial.suggest_float("validation_split", 0.1, 0.5, step=0.1)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

    TRANSFORMER_SETTING["optimizer_choice"] = optimizer
    TRANSFORMER_SETTING["num_heads"] = num_head
    TRANSFORMER_SETTING["head_size"] = head_size
    TRANSFORMER_SETTING["ff_dim"] = ff_dim
    TRANSFORMER_SETTING["num_transformer_blocks"] = num_transformer_blocks
    TRANSFORMER_SETTING["mlp_units"] = mlp_units
    TRANSFORMER_SETTING["dropout"] = dropout
    TRANSFORMER_SETTING["mlp_dropout"] = mlp_dropout
    TRANSFORMER_SETTING["learning_rate"] = learning_rate
    TRANSFORMER_SETTING["validation_split"] = validation_split
    TRANSFORMER_SETTING["print_summary"] = False
    TRANSFORMER_SETTING["batch_size"] = batch_size

    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, **TRANSFORMER_SETTING)
    return transformer.evaluate_model(model, X_test, Y_TEST)


def optimizer():
    study = optuna.create_study(study_name="Transformer Optimization", direction="minimize")
    study.optimize(objective, n_trials=50)
    best_params = study.best_params
    print(f"Best params: {best_params}")


import configparser
import pandas as pd


def generate_config_dic(file_path: str) -> dict:
    config = configparser.ConfigParser()
    config.read(file_path)

    config_dict = {
        # "z_normalize": config.getboolean("normalization", "z_normalize"),
        # 'log_normalize': config.getboolean("normalization", "log_normalize"),
        'normalizer': config.get('normalization', 'normalizer'),
        'use_quantile_filter': config.getboolean("normalization", "use_quantile_filter"),
        "include_std": config.getboolean("general", "include_std"),
        "use_mean_y": config.getboolean("general", "use_mean_y"),
        "window_size": config.getint("general", "window_size"),
        "forecast_size": config.getint("general", "forecast_size"),
        "ma_len": config.getint("general", "ma_len"),
        'quantile_filter': config.getfloat("general", "quantile_filter"),
        "source": config.get("columns", "source"),
        "open_col_name": config.get("columns", "open_col_name"),
        "high_col_name": config.get("columns", "high_col_name"),
        "low_col_name": config.get("columns", "low_col_name"),
        "close_col_name": config.get("columns", "close_col_name"),
        "usable_data_col": eval(config.get("columns", "usable_data_col")),
        "data_folder": Path(config.get("folders", "data_folder")),
        "result_folder": Path(config.get("folders", "result_folder")),
        "models_folder": Path(config.get("folders", "models_folder")),
        "file_name_format": config.get("plots", "file_name_format"),
        "training_cut_off_date": pd.to_datetime(config.get("cut_off_dates", "training_cut_off_date")),
        "file_list": eval(config.get("files", "file_list")),
    }

    return config_dict


def main(
        normalizer: str,
        use_quantile_filter: bool,
        include_std: bool,
        use_mean_y: bool,
        window_size: int,
        forecast_size: int,
        ma_len: int,
        quantile_filter: float,
        source: str,
        open_col_name: str,
        high_col_name: str,
        low_col_name: str,
        close_col_name: str,
        usable_data_col: list,
        data_folder: Path,
        result_folder: Path,
        models_folder: Path,
        file_name_format: str,
        training_cut_off_date: pd.DatetimeTZDtype,
        file_list: list):
    file_name_format = str(eval(file_name_format))
    util = Util(normalizer, file_list, window_size, training_cut_off_date, include_std, source, file_name_format,
                open_col_name, high_col_name, low_col_name, close_col_name, result_folder, data_folder, usable_data_col,
                ma_len, use_mean_y, forecast_size, use_quantile_filter, quantile_filter)

    train_data, test_data = util.gen_multiple_sliding_window()

    X_train, y_train, mean_x_train, std_x_train, mean_y_train, std_y_train, high_train, low_train, close_train, \
        date_train, symbol = train_data

    X_test, y_test, mean_x_test, std_x_test, mean_y_test, std_y_test, high_test, low_test, close_test, date_test, \
        symbol = test_data

    X_train, X_test, y_train, y_test = util.shuffle_reshape(X_train, X_test, y_train, y_test)

    Plotter.plot_hist_y_distribution(y_train, y_test, result_folder)

    history, model = transformer.construct_transformer(X_train=X_train, y_train=y_train, window_size=window_size,
                                                       forecast_size=forecast_size, source=source,
                                                       data_folder=data_folder, normalizer=normalizer,
                                                       model_folder=models_folder, **TRANSFORMER_SETTING)
    # model = transformer.load_model(Path(models_folder / 'daily Window size [20][All Stock].h5'))

    # transformer.evaluate_model(model, X_test, y_test)

    y_pred_normal = model.predict(X_test)
    print(f"converting y_pred_normal to original")
    y_pred = util.convert_to_original(y_pred_normal, mean_y_test, std_y_test)
    print(f"converting y_test to original")
    y_test = util.convert_to_original(y_test, mean_y_test, std_y_test)
    print(f"converting X_test to original")
    X_test = util.convert_to_original(X_test, mean_x_test, std_x_test)

    plotter = Plotter(y_test, y_pred, result_folder, model, file_name_format)

    # Make sure to check the len of y_test
    plotter.plot_scatter_true_vs_predicted(0, len(y_test) // 3)
    plotter.plot_histogram_y_test_minus_y_pred()
    plotter.plot_scatter_true_vs_predicted_diagonal()
    # plotter.plot_scatter_true_vs_predicted_diagonal_only_different_sign()
    temp_res = pd.DataFrame()
    temp_res['date'] = date_test
    temp_res['symbol'] = symbol
    temp_res['open'] = [X_test[i][-1][0] for i in range(len(X_test))]
    temp_res['y_test'] = y_test
    temp_res['y_pred'] = y_pred
    temp_res['high'] = high_test
    temp_res['low'] = low_test
    temp_res['close'] = close_test
    temp_res['open_original'] = [(X_test[i][-1][0]*std_x_test[i]) + mean_x_test[i] for i in range(len(X_test))]
    temp_res['mean_x'] = mean_x_test
    temp_res['std_x'] = std_x_test
    temp_res['mean_y'] = mean_y_test
    temp_res['std_y'] = std_y_test
    temp_res['y_test_original'] = (y_test * std_y_test) + mean_y_test
    temp_res['y_pred_original'] = ((y_pred.flatten()) * std_y_test) + mean_y_test

    temp_res.to_csv(result_folder/"temp_res.csv", index=False)
    results = util.analyze_results(y_test, y_pred, X_test, date_test, high_test, low_test, close_test)
    plotter.plot_cum_log_return(results)


# Call the main function
if __name__ == "__main__":
    config_dictionary = generate_config_dic('config.ini')
    main(**config_dictionary)
    # optimizer()
