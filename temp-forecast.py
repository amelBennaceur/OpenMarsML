# Inspired by: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#  PLotting reference: https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
from datetime import datetime
from math import sqrt
from typing import Callable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import History
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

TRAINING_FLAG_COLUMN: str = "training"


def plot_timeseries(dataframe: pd.DataFrame):
    plt.figure()
    columns: List[str] = dataframe.columns

    for index, column_name in enumerate(columns):
        plt.subplot(len(columns), 1, index + 1)
        plt.title(column_name, y=0.5, loc="right")
        plt.plot(dataframe[column_name])

    plt.show()


def pre_process_data(dataframe: pd.DataFrame, period: str, lag: int) -> Tuple[MinMaxScaler, pd.DataFrame]:
    dataframe = dataframe.resample(period).median()
    min_max_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
    print("Before scaling: dataframe.values.shape ", dataframe.values.shape)
    dataframe.info()
    dataframe[dataframe.columns] = min_max_scaler.fit_transform(dataframe.values)

    past_values: pd.DataFrame = dataframe.shift(lag)
    past_values.columns = ["{}(t - 1)".format(column_name) for column_name in past_values.columns]

    merged_dataframe: pd.DataFrame = pd.concat([past_values, dataframe], axis=1)
    merged_dataframe.dropna(inplace=True)
    merged_dataframe.drop(['Tsurf', 'cloud', 'vapour', 'u_wind', 'v_wind', 'dust', 'temp'], axis=1, inplace=True)

    return min_max_scaler, merged_dataframe


def split_in_train_test(dataframe: pd.DataFrame) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataframe.info()
    print(dataframe.head())

    training_mask: pd.Series = dataframe[TRAINING_FLAG_COLUMN] == True
    training_dataframe: pd.DataFrame = dataframe[training_mask]
    testing_dataframe: pd.DataFrame = dataframe[~training_mask]
    testing_dataframe.drop(TRAINING_FLAG_COLUMN, axis=1, inplace=True)
    training_dataframe.drop(TRAINING_FLAG_COLUMN, axis=1, inplace=True)

    train: np.ndarray = training_dataframe.values
    test: np.ndarray = testing_dataframe.values
    print(f"Training shape: {train.shape}")
    print(f"Testing shape: {test.shape}")

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


def evaluate_and_predict(model: Sequential, min_max_scaler: MinMaxScaler, features: np.ndarray,
                         labels: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    predictions: np.ndarray = model.predict(features)
    features: np.ndarray = features.reshape((features.shape[0], features.shape[2]))
    target_column_index: int = 1

    features[:, target_column_index] = predictions.flatten()
    original_data: np.ndarray = min_max_scaler.inverse_transform(features.copy())
    denormalized_predictions: np.ndarray = original_data[:, target_column_index].copy()

    features[:, target_column_index] = labels.flatten()
    original_data = min_max_scaler.inverse_transform(features.copy())
    denormalized_reals: np.ndarray = original_data[:, target_column_index].copy()

    rmse: float = sqrt(mean_squared_error(denormalized_reals, denormalized_predictions))
    return rmse, denormalized_predictions, denormalized_reals


def load_dataset(training_file: str, testing_file: str) -> pd.DataFrame:
    dataframes: List[pd.DataFrame] = []
    for data_file in [training_file, testing_file]:
        parser: Callable = lambda data_string: datetime.strptime(data_string, '%Y-%m-%d %H:%M:%S')
        dataframe: pd.DataFrame = pd.read_csv(data_file, parse_dates=['Time'],
                                              date_parser=parser, index_col=0)
        print(f"Rows in {data_file}: {len(dataframe)}")
        dataframe.drop(['Ls', 'LT', 'CO2ice'], axis=1, inplace=True)
        dataframe.index.name = "Time"

        if data_file == training_file:
            dataframe[TRAINING_FLAG_COLUMN] = True
        elif data_file == testing_file:
            dataframe[TRAINING_FLAG_COLUMN] = False

        dataframes.append(dataframe)

    return pd.concat(dataframes, axis=0)


def prepare_prediction_data(reference_dataframe: pd.DataFrame, x_start: int, x_end: int,
                            predictions: np.ndarray) -> np.ndarray:
    prediction_data: np.ndarray = np.empty(shape=(len(reference_dataframe), 1))
    prediction_data[:, :] = np.nan
    prediction_data[x_start:x_end, :] = predictions.reshape(predictions.shape + (1,))

    return prediction_data


def main():
    original_dataframe: pd.DataFrame = load_dataset('data/insight_openmars_training_time.csv',
                                                    'data/insight_openmars_test_time.csv')
    print(original_dataframe[TRAINING_FLAG_COLUMN].value_counts())
    plot_timeseries(original_dataframe)

    period: str = "D"
    lstm_neurons: int = 50
    output_neurons: int = 1
    loss_function: str = "mae"
    optimiser: str = "adam"
    epochs: int = 50

    batch_size: int = 72
    lag: int = 1

    min_max_scaler: MinMaxScaler
    scaled_dataframe: pd.DataFrame
    min_max_scaler, scaled_dataframe = pre_process_data(original_dataframe, period, lag)

    train_x: np.ndarray
    train_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    train_x, train_y, test_x, test_y = split_in_train_test(scaled_dataframe)

    print("train_x.shape", train_x.shape)
    print("train_y.shape", train_y.shape)
    print("test_x.shape", test_x.shape)
    print("test_y.shape", test_y.shape)

    model: Sequential = define_model(train_x, lstm_neurons, output_neurons, loss_function, optimiser)
    start_training(model, train_x, train_y, test_x, test_y, epochs, batch_size)

    training_rmse: float
    training_predictions: np.ndarray
    training_denormalized: np.ndarray

    training_rmse, training_predictions, training_denormalized = evaluate_and_predict(model, min_max_scaler, train_x,
                                                                                      train_y)
    print("Training RMSE: {}".format(training_rmse))

    test_rmse: float
    testing_predictions: np.ndarray
    testing_denormalized: np.ndarray
    test_rmse, testing_predictions, testing_denormalized = evaluate_and_predict(model, min_max_scaler, test_x, test_y)
    print("Test RMSE: {}".format(test_rmse))

    training_prediction_data: np.ndarray = prepare_prediction_data(scaled_dataframe, 0,
                                                                   len(training_predictions),
                                                                   training_predictions)
    testing_prediction_data: np.ndarray = prepare_prediction_data(scaled_dataframe,
                                                                  len(training_predictions),
                                                                  len(training_predictions) + len(
                                                                      testing_predictions),
                                                                  testing_predictions)

    line_width: float = 0.3
    plt.plot(np.concatenate([training_denormalized, testing_denormalized]), color="blue", linewidth=line_width)
    plt.plot(training_prediction_data, color="green", linewidth=line_width)
    plt.plot(testing_prediction_data, color="red", linewidth=line_width)

    plt.show()


if __name__ == "__main__":
    main()
