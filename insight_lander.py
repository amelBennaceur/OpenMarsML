from datetime import timedelta
from typing import Dict, List, Tuple

import pandas as pd
import matplotlib.pyplot as plt
from gluonts.dataset.common import ListDataset


def get_dataset() -> pd.DataFrame:
    dataset: pd.DataFrame = pd.read_csv("data/insight_openmars_withobs.csv", header=0, parse_dates=[0],
                                        index_col=0, na_values="-9999")
    column_map: Dict[str, str] = {"Ls": "solar_longitude",
                                  "LT": "local_time",
                                  "Psurf_assim": "assim_surface_pressure",
                                  "Psurf_obs": "surface_pressure",
                                  "u_assim": "assim_eastwind_speed",
                                  "u_obs": "eastward_wind_speed",
                                  "v_assim": "assim_northwind_speed",
                                  "v_obs": "northward_wind_speed",
                                  "dust_assim": "assim_dust_opticaldepth",
                                  "temp_assim": "assim_air_temperature",
                                  "temp_obs": "air_temperature"}

    dataset = dataset.rename(columns=column_map, errors="raise")
    dataset.index = dataset.index.rename("observation_time")

    dataset = dataset.replace(-9999, None)

    dataset = dataset.interpolate(method="linear", axis=0)

    return dataset


def extract_time_series(dataframe: pd.DataFrame, column_name: str) -> pd.Series:
    return dataframe[column_name]


def get_target(target: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.Series:
    training_range = target.index[(target.index > start_date) & (target.index < end_date)]
    return target[training_range]


def to_list_dataset(target: pd.Series, start_date: pd.Timestamp, end_date: pd.Timestamp,
                    frequency: str) -> Tuple[pd.Series, ListDataset]:
    target_series: pd.Series = get_target(target, start_date, end_date)
    list_dataset: ListDataset = ListDataset([{"target": target_series,
                                              "start": str(start_date)}], freq=frequency)
    return target_series, list_dataset


def plot_time_series(dataframe: pd.DataFrame, column_names: List[str]):
    figures, axis = plt.subplots(nrows=len(column_names), ncols=1, figsize=(20, 20), sharex="all")
    grid = axis.ravel()
    for plot_index in range(0, len(column_names)):
        time_series: pd.Series = extract_time_series(dataframe, column_names[plot_index])
        time_series.plot(ax=grid[plot_index])
        grid[plot_index].set_xlabel("Observation_Time")
        grid[plot_index].set_ylabel(time_series.name)
        grid[plot_index].grid(which="minor", axis="x")

    return None
