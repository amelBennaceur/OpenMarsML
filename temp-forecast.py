# Inspired by: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/
#  PLotting reference:
#  https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
from datetime import datetime
from math import sqrt
from typing import Callable, List, Tuple, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from keras.callbacks import History
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

FEATURE_COLUMNS: List[str] = ['Tsurf', 'cloud', 'vapour', 'u_wind', 'v_wind', 'dust', 'temp']
EXCLUDED_COLUMNS: List[str] = ['Ls', 'LT', 'CO2ice']
TARGET_COLUMN: str = 'Psurf'

TRAINING_FLAG_COLUMN: str = "training"
TRAINING_FILE: str = 'data/insight_openmars_training_time.csv'
TESTING_FILE: str = 'data/insight_openmars_test_time.csv'
# EPOCHS: int = 50


EPOCHS: int = 3


def plot_timeseries(dataframe: pd.DataFrame):
    plt.figure()
    columns: List[str] = dataframe.columns

    for index, column_name in enumerate(columns):
        plt.subplot(len(columns), 1, index + 1)
        plt.title(column_name, y=0.5, loc="right")
        plt.plot(dataframe[column_name])

    plt.show()


def pre_process_data(dataframe: pd.DataFrame, period: str, lag: int) -> Tuple[MinMaxScaler, pd.DataFrame]:
    if period:
        dataframe = dataframe.resample(period).median()
    min_max_scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1))

    columns_for_scaling: List[str] = FEATURE_COLUMNS + [TARGET_COLUMN]
    data_for_scaling: pd.DataFrame = dataframe[columns_for_scaling]
    print("Before scaling: data_for_scaling.values.shape ", data_for_scaling.values.shape)
    data_for_scaling.info()

    dataframe[columns_for_scaling] = min_max_scaler.fit_transform(data_for_scaling)

    past_values: pd.DataFrame = dataframe.shift(lag)
    past_values.columns = ["{}(t - 1)".format(column_name) for column_name in past_values.columns]

    merged_dataframe: pd.DataFrame = pd.concat([past_values, dataframe], axis=1)
    merged_dataframe.dropna(inplace=True)

    merged_columns: List[str] = [f"{feature_column}(t - 1)" for feature_column in columns_for_scaling] + [
        TRAINING_FLAG_COLUMN] + [TARGET_COLUMN]
    merged_dataframe = merged_dataframe[merged_columns]

    return min_max_scaler, merged_dataframe


def split_in_train_test(dataframe: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    training_mask: pd.Series = dataframe[TRAINING_FLAG_COLUMN] == True
    training_dataframe: pd.DataFrame = dataframe[training_mask]
    testing_dataframe: pd.DataFrame = dataframe[~training_mask]
    testing_dataframe.drop(TRAINING_FLAG_COLUMN, axis=1, inplace=True)
    training_dataframe.drop(TRAINING_FLAG_COLUMN, axis=1, inplace=True)

    train: np.ndarray = training_dataframe.values
    print(f"Training shape: {train.shape}")
    print(training_dataframe.head())

    test: np.ndarray = testing_dataframe.values
    print(f"Testing shape: {test.shape}")
    print("Training dataframe: ")
    testing_dataframe.info()

    train_x: np.ndarray = train[:, :-1]
    train_y: np.ndarray = train[:, -1]
    test_x: np.ndarray = test[:, :-1]
    test_y: np.ndarray = test[:, -1]

    train_x = train_x.reshape((train_x.shape[0], 1, train_x.shape[1]))
    test_x = test_x.reshape((test_x.shape[0], 1, test_x.shape[1]))

    return train_x, train_y, test_x, test_y


def define_model(train_x: np.ndarray, lstm_neurons: int, output_neurons: int, loss_function: str,
                 optimiser: str) -> keras.Sequential:
    model: keras.Sequential = keras.Sequential()
    model.add(layers.LSTM(lstm_neurons, input_shape=(train_x.shape[1], train_x.shape[2])))
    model.add(layers.Dense(output_neurons))
    model.compile(loss=loss_function, optimizer=optimiser)

    return model


def start_training(model: keras.Sequential, train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray,
                   test_y: np.ndarray, epochs: int, batch_size: int):
    history: History = model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                                 validation_data=(test_x, test_y),
                                 verbose=2, shuffle=False)

    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def evaluate_and_predict(model: keras.Sequential, min_max_scaler: MinMaxScaler, features: np.ndarray,
                         labels: np.ndarray, prediction_file: Optional[str] = None) -> Tuple[
    float, np.ndarray, np.ndarray]:
    predictions: np.ndarray = model.predict(features)
    features: np.ndarray = features.reshape((features.shape[0], features.shape[2]))
    print(f"predictions.shape {predictions.shape}")
    print(f"features.shape {features.shape}")
    target_column_index: int = len(FEATURE_COLUMNS)

    features[:, target_column_index] = predictions.flatten()
    original_data: np.ndarray = min_max_scaler.inverse_transform(features.copy())
    if prediction_file:
        np.savetxt(prediction_file, original_data, delimiter=",")
        print(f"File {prediction_file} created")

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
        dataframe.drop(EXCLUDED_COLUMNS, axis=1, inplace=True)
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


def append_predictions(original_file: str, predictions: np.ndarray, predicted_column: str):
    data_from_file: pd.DataFrame = pd.read_csv(original_file)
    data_from_file[f'{predicted_column}(Pred)'] = predictions
    filename: str = f'{original_file}_{predicted_column}_pred.csv'
    data_from_file.to_csv(filename, index=False)
    print(f"{filename} created.")


def main():
    original_dataframe: pd.DataFrame = load_dataset(TRAINING_FILE,
                                                    TESTING_FILE)
    print(original_dataframe[TRAINING_FLAG_COLUMN].value_counts())
    plot_timeseries(original_dataframe)

    # period: str = "D"
    # Without daily median.
    period: str = ""
    lstm_neurons: int = 50
    output_neurons: int = 1
    loss_function: str = "mae"
    optimiser: str = "adam"
    epochs: int = EPOCHS

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

    model: keras.Sequential = define_model(train_x, lstm_neurons, output_neurons, loss_function, optimiser)
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
    test_rmse, testing_predictions, testing_denormalized = evaluate_and_predict(model, min_max_scaler, test_x, test_y,
                                                                                prediction_file="pred_on_test.csv")
    print("Test RMSE: {}".format(test_rmse))
    append_predictions(TESTING_FILE, testing_predictions, TARGET_COLUMN)

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
