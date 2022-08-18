import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import optuna
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.model import Predictor, Forecast
from optuna import Trial, Study
from optuna.trial import FrozenTrial

from deepar_forecast import train_predictor

plt.rcParams.update({'font.size': 10})


class DeepARTuningObjective:

    def __init__(self, learning_rate: float, num_batches_per_epoch: int, frequency: str,
                 metric_type: str, training_dataset: ListDataset):
        self.training_epochs: int = 10
        self.learning_rate: float = learning_rate
        self.num_batches_per_epoch: int = num_batches_per_epoch
        self.frequency: str = frequency
        self.metric_type: str = metric_type

        self.training_dataset: ListDataset = training_dataset

    @staticmethod
    def get_params(trial: Trial) -> dict:
        return {
            "context_length": trial.suggest_categorical("context_length",
                                                        [num_days * 12 for num_days in range(1, 6)]),
            "batch_size": trial.suggest_categorical("batch_size",
                                                    [32, 64, 128, 256, 512])
        }

    def __call__(self, trial: Trial):
        parameters: dict = self.get_params(trial)
        entry_split: List[Tuple[dict, pd.DataFrame]] = [self.split_entry(entry, parameters["context_length"])
                                                        for entry in self.training_dataset]
        entry_pasts: List[dict] = [entry[0] for entry in entry_split]
        entry_futures: List[pd.DataFrame] = [entry[1] for entry in entry_split]

        predictor: Predictor = train_predictor(training_dataset=entry_pasts,
                                               prediction_length=parameters["context_length"],
                                               context_length=parameters["context_length"],
                                               epochs=self.training_epochs,
                                               learning_rate=self.learning_rate,
                                               num_batches_per_epoch=self.num_batches_per_epoch,
                                               frequency=self.frequency,
                                               batch_size=parameters["batch_size"],
                                               save=False)

        forecast_iterator = predictor.predict(entry_pasts)
        forecast_list: List[Forecast] = list(forecast_iterator)
        evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
        aggregate_metrics, item_metrics = evaluator(entry_futures, forecast_list)

        return aggregate_metrics[self.metric_type]

    def split_entry(self, entry: dict, prediction_length=int) -> Tuple[dict, pd.DataFrame]:
        entry_past: dict = {}
        for key, value in entry.items():
            if key == "target":
                entry_past[key] = value[: -prediction_length]
            else:
                entry_past[key] = value

        entry_future: pd.DataFrame = pd.DataFrame(entry["target"], columns=[entry["item_id"]],
                                                  index=pd.period_range(start=entry["start"],
                                                                        periods=len(entry["target"]),
                                                                        freq=self.frequency))

        return entry_past, entry_future[-prediction_length:]


def get_dataset() -> pd.DataFrame:
    dataset: pd.DataFrame = pd.read_csv("data/insight_openmars_withobs.csv", header=0, parse_dates=[0],
                                        index_col=0, na_values="-9999")
    column_map: Dict[str, str] = {"Sol": "martian_days",
                                  "Ls": "solar_longitude",
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

    print("Missing values")
    print(dataset.isna().sum())

    dataset = dataset.replace(-9999, None)

    dataset = dataset.interpolate(method="linear", axis=0)

    return dataset


def optimise_parameters(number_of_trials: int, learning_rate: float, num_batches_per_epoch: int, frequency: str,
                        metric_type: str, training_dataset: ListDataset) -> FrozenTrial:
    start_time: float = time.time()

    study: Study = optuna.create_study(direction="minimize")
    study.optimize(DeepARTuningObjective(learning_rate=learning_rate,
                                         num_batches_per_epoch=num_batches_per_epoch,
                                         frequency=frequency,
                                         metric_type=metric_type,
                                         training_dataset=training_dataset), n_trials=number_of_trials)

    print(f"Finished {len(study.trials)}")

    trial: FrozenTrial = study.best_trial
    print(f"Best trial value {trial.value}")
    print(f"Best trial params: {trial.params}")
    print(f"Optimization time: {time.time() - start_time} s")

    return trial


def extract_time_series(dataframe: pd.DataFrame, column_name: str, index_columns: List[str]) -> pd.Series:
    indexed_dataframe: pd.DataFrame = dataframe
    if len(index_columns) > 0:
        indexed_dataframe: pd.DataFrame = dataframe.set_index(index_columns)
    return indexed_dataframe[column_name]


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


def plot_time_series(dataframe: pd.DataFrame, column_names: List[str], index_columns: List[str]):
    figures, axis = plt.subplots(nrows=len(column_names), ncols=1, figsize=(20, 20), sharex="all")
    grid = axis.ravel()
    for plot_index in range(0, len(column_names)):
        time_series: pd.Series = extract_time_series(dataframe, column_names[plot_index], index_columns)
        time_series.plot(ax=grid[plot_index])
        grid[plot_index].set_xlabel("Time")
        grid[plot_index].set_ylabel(time_series.name)
        grid[plot_index].grid(which="minor", axis="x")

    return None
