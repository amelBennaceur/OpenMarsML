import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd

def plot_forecast(
    input_dataframe: TimeSeriesDataFrame, 
    expected_dataframe:TimeSeriesDataFrame,
    forecast_dataframe: TimeSeriesDataFrame,
    time_series_id:str,
    prediction_length: int,
    training_values: int = 200,
    figure_size: tuple[int, int] = (20, 3)) -> None:

    plt.figure(figsize=figure_size)

    past_values: pd.Series = input_dataframe.loc[time_series_id]["target"]
    predicted_values: pd.Series = forecast_dataframe.loc[time_series_id]
    expected_values: pd.Series = expected_dataframe.loc[time_series_id][-prediction_length:]

    plt.plot(past_values[-training_values:], label=f"Past {time_series_id} values")
    plt.plot(predicted_values["mean"], label="Mean forecast")
    plt.plot(expected_values, label=f"Future {time_series_id} values")

    plt.fill_between(
        predicted_values.index, predicted_values["0.1"], predicted_values["0.9"],
        color="red", alpha=0.1, label=f"10%-90% confidence interval"
    )

    plt.legend()