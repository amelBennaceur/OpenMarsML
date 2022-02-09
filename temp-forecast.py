from datetime import datetime
from math import sqrt
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import Sequential
from keras.callbacks import History
from keras.layers import LSTM, Dense
from numpy import concatenate
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


def plot_timeseries(dataframe: pd.DataFrame):
    plt.figure()
    columns: List[str] = dataframe.columns

    for index, column_name in enumerate(columns):
        plt.subplot(len(columns), 1, index + 1)
        plt.title(column_name, y=0.5, loc="right")
        plt.plot(dataframe[column_name])

    plt.show()


def pre_process_data(dataframe: pd.DataFrame, period: str) -> Tuple[MinMaxScaler, pd.DataFrame]:
    dataframe = dataframe.resample(period).median()
    min_max_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    print("Before scaling: dataframe.values.shape ", dataframe.values.shape)
    dataframe.info()
    dataframe[dataframe.columns] = min_max_scaler.fit_transform(dataframe.values)

    lag: int = 1
    past_values: pd.DataFrame = dataframe.shift(lag)
    past_values.columns = ["{}(t - 1)".format(column_name) for column_name in past_values.columns]

    merged_dataframe: pd.DataFrame = pd.concat([past_values, dataframe], axis=1)
    merged_dataframe.dropna(inplace=True)
    merged_dataframe.drop(['Tsurf', 'cloud', 'vapour', 'u_wind', 'v_wind', 'dust', 'temp'], axis=1, inplace=True)

    return min_max_scaler, merged_dataframe


def split_in_train_test(dataframe: pd.DataFrame, train_days: int) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataframe.info()
    print(dataframe.head())

    raw_values: np.ndarray = dataframe.values
    train: np.ndarray = raw_values[:train_days, :]
    test: np.ndarray = raw_values[train_days:, :]

    train_x: np.ndarray = train[:, :-1]
    train_y: np.ndarray = train[:, -1]
    test_x: np.ndarray = test[:, :-1]
    test_y: np.ndarray = test[:, -1]

    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    return train_x, train_y, test_x, test_y


def define_model(train_x: np.ndarray, lstm_neurons: int, output_neurons: int, loss_function: str,
                 optimiser: str) -> Sequential:
    model: Sequential = Sequential()
    model.add(LSTM(lstm_neurons, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(Dense(output_neurons))
    model.compile(loss=loss_function, optimizer=optimiser)

    return model


def start_training(model: Sequential, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                   test_y: np.ndarray, epochs: int, batch_size: int):
    history: History = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                                 validation_data=(test_x, test_y),
                                 verbose=2, shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def evaluate(model: Sequential, min_max_scaler: MinMaxScaler, test_x: np.ndarray, test_y: np.ndarray) -> float:
    predictions: np.ndarray = model.predict(test_x)
    print("predictions.shape", predictions.shape)
    test_x: np.ndarray = test_x.reshape((test_x.shape[0], test_x.shape[2]))
    target_column_index: int = 1

    print("Before re-scaling: original_data.shape", test_x.shape)
    test_x[:, target_column_index] = predictions.flatten()
    original_data: np.ndarray = min_max_scaler.inverse_transform(test_x)
    denormalized_predictions: np.ndarray = original_data[:, target_column_index]

    test_x[:, target_column_index] = test_y.flatten()
    original_data = min_max_scaler.inverse_transform(test_x)
    denormalized_reals: np.ndarray = original_data[:, target_column_index]

    rmse: float = sqrt(mean_squared_error(denormalized_reals, denormalized_predictions))
    return rmse


def main():
    parser: Callable = lambda data_string: datetime.strptime(data_string, '%Y-%m-%d %H:%M:%S')
    dataframe: pd.DataFrame = pd.read_csv('data/insight_openmars_training_time.csv', parse_dates=['Time'],
                                          date_parser=parser, index_col=0)
    dataframe.drop(['Ls', 'LT', 'CO2ice'], axis=1, inplace=True)
    dataframe.index.name = "Time"
    # plot_timeseries(dataframe)

    period: str = "D"
    train_days: int = 365 * 10
    lstm_neurons: int = 50
    output_neurons: int = 1
    loss_function: str = "mae"
    optimiser: str = "adam"
    epochs: int = 50
    batch_size: int = 72

    min_max_scaler: MinMaxScaler
    dataframe: pd.DataFrame
    min_max_scaler, dataframe = pre_process_data(dataframe, period)

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_x, train_y, test_x, test_y = split_in_train_test(dataframe, train_days)

    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    print("test_x.shape", test_x.shape)
    print("test_y.shape", test_y.shape)

    model: Sequential = define_model(train_x, lstm_neurons, output_neurons, loss_function, optimiser)
    start_training(model, train_x, train_y, test_x, test_y, epochs, batch_size)

    rmse: float = evaluate(model, min_max_scaler, test_x, test_y)
    print("Test RMSE: {}".format(rmse))


if __name__ == "__main__":
    main()
