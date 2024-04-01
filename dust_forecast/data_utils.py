import pandas as pd
import logging

DATETIME_COLUMN: str = "Time"
METRIC_COLUMN: str = "Metric"
VALUE_COLUMN: str = "target"

def to_wide_format(dataframe: pd.DataFrame, time_column:str) -> pd.DataFrame:
    columns: pd.Index = dataframe.columns

    measure_dataframes: list[pd.DataFrame] = []
    for column in columns:
        if column != time_column:
            time_and_value: pd.DataFrame = dataframe.loc[:, [time_column, column]] # type: ignore
            time_and_value[METRIC_COLUMN] = column
            time_and_value = time_and_value.rename(columns={column: VALUE_COLUMN})

            logging.info(f"Metric: {column} Time steps: {len(time_and_value.index)}")

            measure_dataframes.append(time_and_value)

    logging.info(f"Number of metrics: {len(measure_dataframes)}")
    result_dataframe: pd.DataFrame = pd.concat(measure_dataframes, axis=0)

    return result_dataframe