from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
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


def to_list_dataset(targets: List[pd.Series], start_date: pd.Timestamp, end_date: pd.Timestamp,
                    frequency: str) -> Tuple[List, ListDataset]:
    target_list: List[Dict] = [{"target": get_target(current_target, start_date, end_date),
                                "start": str(start_date),
                                "item_id": current_target.name}
                               for current_target in targets]
    list_dataset: ListDataset = ListDataset(target_list, freq=frequency)
    return target_list, list_dataset


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
