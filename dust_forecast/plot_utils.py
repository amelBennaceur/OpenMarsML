import matplotlib.pyplot as plt
from autogluon.timeseries import TimeSeriesDataFrame
import pandas as pd
from typing import Optional

def plot_forecast(
    input_dataframe: TimeSeriesDataFrame, 
    expected_dataframe:TimeSeriesDataFrame,
    time_series_id:str,
    prediction_length: int,
    forecast_dataframe: Optional[TimeSeriesDataFrame]=None,
    training_values: int = 200,
    figure_size: tuple[int, int] = (20, 3)) -> None:

    plt.figure(figsize=figure_size)

    past_values: pd.Series = input_dataframe.loc[time_series_id]["target"]
    plt.plot(past_values[-training_values:], label=f"Past {time_series_id} values")

    if forecast_dataframe is not None:
        predicted_values: pd.Series = forecast_dataframe.loc[time_series_id]
        plt.plot(predicted_values["mean"], label="Mean forecast")
        plt.fill_between(
            predicted_values.index, predicted_values["0.1"], predicted_values["0.9"],
            color="red", alpha=0.1, label=f"10%-90% confidence interval"
        )
    else:
        plt.fill_betweenx(
            y=(0, expected_dataframe.loc[time_series_id]["target"].max()), 
            # y=(0, 2), 
            x1=expected_dataframe.loc[time_series_id].index.max(), 
            x2=input_dataframe.loc[time_series_id].index.max(),
            color="red", alpha=0.1, label=f"test forecast horizon",
        )

    expected_values: pd.Series = expected_dataframe.loc[time_series_id][-prediction_length:]
    plt.plot(expected_values, label=f"Future {time_series_id} values")


    plt.grid(which="both")
    plt.xlabel("Timestamp")
    plt.ylabel("Value")
    plt.title(f"{time_series_id} forecast")

    plt.legend(loc="upper left")

def plot_time_series(
        time_series_dataframe: TimeSeriesDataFrame,
        time_series_ids: list[str], figure_size=(15, 8)) -> None:
    
    plt.figure(figsize=figure_size)
    for time_series_id in time_series_ids:
        time_series_to_plot: pd.Series = time_series_dataframe.loc[time_series_id]
        plt.plot(time_series_to_plot, label=time_series_id)
    plt.legend()
