from typing import List

# Evaluation reference: https://aws.amazon.com/blogs/machine-learning/creating-neural-time-series-models-with-gluon-time-series/
# Training reference: https://learning.oreilly.com/library/view/advanced-forecasting-with/9781484271506/html/508548_1_En_20_Chapter.xhtml

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
import pandas as pd
from gluonts.dataset.common import ListDataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.forecast import Forecast
from gluonts.model.predictor import Predictor
from gluonts.mx.trainer import Trainer

import tempforecast

SEED: int = 7


def create_list_dataset(filename: str, frequency: str) -> ListDataset:
    dataframe: pd.DataFrame = tempforecast.csv_to_dataframe(filename)
    print("dataframe.describe()", dataframe.describe())

    target: pd.DataFrame = dataframe[tempforecast.TARGET_COLUMN]
    print("target.head()", target.head())
    start: pd.Timestamp = dataframe.index[0]
    print("start", start)

    list_dataset: ListDataset = ListDataset([{'target': target,
                                              'start': start}],
                                            freq=frequency)
    print("list_dataset", list_dataset)
    return list_dataset


def train_predictor(training_dataset: ListDataset, epochs: int, learning_rate: float, num_batches_per_epoch: int,
                    prediction_length: int, context_length: int, frequency: str) -> Predictor:
    trainer: Trainer = Trainer(epochs=epochs,
                               learning_rate=learning_rate,
                               num_batches_per_epoch=num_batches_per_epoch)
    deep_ar_estimator: DeepAREstimator = DeepAREstimator(prediction_length=prediction_length,
                                                         context_length=context_length,
                                                         freq=frequency,
                                                         trainer=trainer)
    predictor: Predictor = deep_ar_estimator.train(training_dataset)

    return predictor


def plot_forecasts(actual: pd.Series, forecast: Forecast, past_length: int):
    _ = actual[-past_length:].plot(figsize=(12, 5), linewidth=2)
    forecast.plot(color='g')
    plt.grid(which='both')
    plt.legend(["observations", "median prediction", "90% confidence interval", "50% confidence interval"])
    plt.show()


def main():
    frequency: str = "2h"
    training_dataset: ListDataset = create_list_dataset(tempforecast.TRAINING_FILE, frequency)

    np.random.seed(SEED)
    mx.random.seed(SEED)

    prediction_length: int = 2 * 12
    context_length: int = prediction_length * 7
    epochs: int = 10
    # epochs: int = 2
    learning_rate: float = 1e-3
    num_batches_per_epoch: int = 100
    evaluation_samples: int = 100

    predictor: Predictor = train_predictor(training_dataset=training_dataset,
                                           prediction_length=prediction_length,
                                           context_length=context_length,
                                           epochs=epochs,
                                           learning_rate=learning_rate,
                                           num_batches_per_epoch=num_batches_per_epoch,
                                           frequency=frequency)

    testing_dataset: ListDataset = create_list_dataset(tempforecast.TESTING_FILE, frequency)

    forecast_iterator, actual_iterator = make_evaluation_predictions(dataset=testing_dataset, predictor=predictor,
                                                                     num_samples=evaluation_samples)
    forecast_list: List[Forecast] = list(forecast_iterator)
    actual_list: List[pd.Series] = list(actual_iterator)
    plot_forecasts(actual_list[0], forecast_list[0], past_length=context_length)

    evaluator: Evaluator = Evaluator(quantiles=[0.5])
    aggregate_metrics, item_metrics = evaluator(iter(actual_list), iter(forecast_list))
    print("aggregate_metrics", aggregate_metrics)


if __name__ == "__main__":
    main()
