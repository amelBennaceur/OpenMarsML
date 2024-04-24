# %%
import sys
import torch
import mlflow
import optuna
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from darts import TimeSeries
import plotly.express as px
from plotly import graph_objects as go
import matplotlib.pyplot as plt
from darts.metrics import mae, mse, mape, rmse, smape
from darts.dataprocessing.transformers.scaler import Scaler
from pytorch_lightning.callbacks import EarlyStopping
from optuna.integration import PyTorchLightningPruningCallback
from darts.models import TiDEModel, BlockRNNModel, TCNModel, TransformerModel, TCNModel, NBEATSModel

sys.path.append('../utils/')

from config import config

freq='2H3T14S'
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
mlflow.set_experiment("OpenMarsHyperOpt")
pd.options.plotting.backend = "plotly"


# %%
def load_dataset(training_file, testing_file):
    dataframes = []
    for data_file in [training_file, testing_file]:
        parser = lambda data_string: datetime.strptime(data_string, '%Y-%m-%d %H:%M:%S')
        dataframe = pd.read_csv(data_file, parse_dates=['Time'],
                                date_parser=parser)
        print(f"Rows in {data_file}: {len(dataframe)}")
        dataframe.drop(['Ls', 'LT', 'CO2ice'], axis=1, inplace=True)
        dataframes.append(dataframe)

    return pd.concat(dataframes, axis=0)


def preprocess(dataframe):
        time = pd.date_range("1998-07-15 21:23:39", periods=len(dataframe), freq=freq)
        dataframe.index = time
        dataframe = dataframe.drop(['Time'], axis=1)
        return dataframe

def create_series(dataframe):
        series = TimeSeries.from_dataframe(dataframe, time_col=None, value_cols=None, fill_missing_dates=True, freq='7394S', fillna_value=None)
        return series.astype(np.float32)

def create_train_val_test_series(series):
        train, temp = series.split_after(0.7)
        val, test = temp.split_after(0.67)
        return train, val, test

# %%
dataframe = load_dataset('../data/data_files/insight_openmars_training_time.csv',
                         '../data/data_files/insight_openmars_test_time.csv')
dataframe = preprocess(dataframe)
train, val, test = create_train_val_test_series(create_series(dataframe))
# print(len(train), len(val), len(test))

# %%
val['dust'].plot(label='val')
test['dust'].plot(label='test')
train['dust'].plot(label='train')


# %%
scaler = Scaler()  # default uses sklearn's MinMaxScaler
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
models = {}


def create_tide_model(trial):

    # select input and output chunk lengths
    in_len = 84
    out_len =  12
    batch_size = trial.suggest_categorical('batch_size', [32, 96, 128, 256])

    # Other hyperparameters
    num_encoder_layers = trial.suggest_categorical("num_encoder_layers", [1, 2])
    num_decoder_layers =  trial.suggest_categorical("num_decoder_layers", [1, 2])
    decoder_output_dim = trial.suggest_categorical("decoder_output_dim",[8, 16])
    hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256])
    temporal_decoder_hidden = trial.suggest_categorical("temporal_decoder_hidden", [16, 32, 64])
    use_layer_norm = trial.suggest_categorical('use_layer_norm', [True, False])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # reproducibility
    torch.manual_seed(42)

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.0008, patience=3, verbose=False)
    callbacks = [pruner, early_stopper]

    pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 50,
    "accelerator": "auto",
    "callbacks": callbacks,
    }

    common_model_args = {
        "optimizer_kwargs": {'lr': lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "random_state": 42,

    }
    # build the BlockRNNModel model
    model = TiDEModel(input_chunk_length= in_len,
                       output_chunk_length = out_len,
                       num_encoder_layers = num_encoder_layers,
                       num_decoder_layers = num_decoder_layers,
                       decoder_output_dim = decoder_output_dim,
                       hidden_size = hidden_size,
                       temporal_decoder_hidden = temporal_decoder_hidden,
                       use_layer_norm = use_layer_norm,
                       use_reversible_instance_norm =True,
                       batch_size = batch_size,
                       model_name="TiDEModel_84_12_HyperOpt", 
                       dropout = dropout,
                       **common_model_args)
    return model

# %%
def create_block_rnn_model(trial):

    # select input and output chunk lengths
    in_len = 84
    out_len =  12
    batch_size = trial.suggest_categorical('batch_size', [32, 96, 128, 256])

    # Other hyperparameters
    n_rnn_layers = trial.suggest_int("n_rnn_layers", 2, 5)
    hidden_dim =  trial.suggest_categorical("hidden_dim", [20, 30, 40, 60])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # reproducibility
    torch.manual_seed(42)

    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.0008, patience=3, verbose=False)
    callbacks = [pruner, early_stopper]

    pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 50,
    "accelerator": "auto",
    "callbacks": callbacks,
    }

    common_model_args = {
        "optimizer_kwargs": {'lr': lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "random_state": 42,

    }
    # build the BlockRNNModel model
    model = BlockRNNModel(model = "LSTM",
                                input_chunk_length= in_len,
                                output_chunk_length = out_len,
                                n_rnn_layers = n_rnn_layers,
                                hidden_dim = hidden_dim,
                                batch_size = batch_size,
                                model_name="BlockRNNModel_84_12_HyperOpt", 
                                dropout = dropout,
                                **common_model_args)
    return model


def create_tcn_model(trial):
    in_len = 84
    out_len =  12
    batch_size = trial.suggest_categorical('batch_size', [32, 96, 128, 256])
    # Other hyperparameters
    kernel_size = trial.suggest_int("kernel_size", 2, 5)
    num_filters = trial.suggest_int("num_filters", 2, 6)
    weight_norm = trial.suggest_categorical("weight_norm", [False, True])
    dilation_base = trial.suggest_int("dilation_base", 2, 4)
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # reproducibility
    torch.manual_seed(42)


    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.0008, patience=2, verbose=False)
    callbacks = [pruner, early_stopper]



    pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 50,
    "accelerator": "auto",
    "callbacks": callbacks,
    }

    common_model_args = {
        "optimizer_kwargs": {'lr': lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,

    }

    # build the TCN model
    model = TCNModel(
    input_chunk_length= in_len,
    output_chunk_length = out_len,
    dilation_base = dilation_base,
    weight_norm = weight_norm,
    kernel_size = kernel_size,
    num_filters = num_filters,
    model_name = 'TCNModel_84_12_HyperOpt',
    dropout = dropout,
    **common_model_args
    )

    return model

def create_transformer_model(trial):
    in_len = 84
    out_len =  12
    batch_size = trial.suggest_categorical('batch_size', [32, 96, 128, 256])

    # Other hyperparameters

    d_model=trial.suggest_categorical('d_model', [8,12,16])
    if d_model == 8:
        nhead=4
    elif d_model == 12:
        nhead=6
    elif d_model == 16:
        nhead=8
    num_encoder_layers=trial.suggest_categorical('num_encoder_layers', [2,4])
    num_decoder_layers=trial.suggest_categorical('num_decoder_layers', [2,4])
    dim_feedforward=trial.suggest_categorical('dim_feedforward', [64, 128, 256])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # reproducibility
    torch.manual_seed(42)


    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.0008, patience=2, verbose=False)
    callbacks = [pruner, early_stopper]



    pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 50,
    "accelerator": "auto",
    "callbacks": callbacks,
    }

    common_model_args = {
        "optimizer_kwargs": {'lr': lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,

    }

    # build the NBetas model
    model = TransformerModel(
    input_chunk_length=in_len,
    output_chunk_length=out_len,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    num_decoder_layers=num_decoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation="relu",
    model_name = 'TransformerModel_84_12_HyperOpt',
    **common_model_args,
)
    
    return model

def objective_tide(trial):
    # select input and output chunk lengths
    model = create_tide_model(trial)
        # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0

    series = create_series(dataframe)

    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = val

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    models[model.model_name] = TiDEModel.load_from_checkpoint("TiDEModel_84_12_HyperOpt", best=True)

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.historical_forecasts(series=val, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=True, 
                                        forecast_horizon=model.model_params['output_chunk_length']
                                        )
    smapes = rmse(val['dust'], preds['dust'], n_jobs=-1, verbose=True)
    return smapes if smapes != np.nan else float("inf")



# %%

def objective_tcn(trial):
    # select input and output chunk lengths
    model = create_tcn_model(trial)
        # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0

    series = create_series(dataframe)

    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = val

    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    models[model.model_name] = TCNModel.load_from_checkpoint("TCNModel_84_12_HyperOpt", best=True)

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.historical_forecasts(series=val, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=True, 
                                        forecast_horizon=model.model_params['output_chunk_length']
                                        )
    smapes = rmse(val['dust'], preds['dust'], n_jobs=-1, verbose=True)
    # smape_val = np.mean(smapes)
    # smape_val = smapes

    return smapes if smapes != np.nan else float("inf")


def objective_block_rnn(trial):
    model = create_block_rnn_model(trial)
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
        # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0
    model_val_set = val
    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    model = BlockRNNModel.load_from_checkpoint("BlockRNNModel_84_12_HyperOpt", best=True)

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.historical_forecasts(series=val, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=True, 
                                        forecast_horizon=model.model_params['output_chunk_length']
                                        )
    smapes = rmse(val['dust'], preds['dust'], n_jobs=-1, verbose=True)
    smape_val = smapes

    return smape_val if smape_val != np.nan else float("inf")

def create_nbeats_model(trial):
    # select input and output chunk lengths
    in_len = 84
    out_len =  12
    batch_size = trial.suggest_categorical('batch_size', [32, 96, 256])

    # Other hyperparameters

    num_blocks=trial.suggest_int('num_blocks', 2, 4)
    num_layers=trial.suggest_int('num_layers', 2, 5)
    layer_widths=trial.suggest_categorical('layer_widths', [256,512])
    dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2, 0.25])
    lr = trial.suggest_float("lr", 5e-5, 1e-3, log=True)
    # reproducibility
    torch.manual_seed(42)


    pruner = PyTorchLightningPruningCallback(trial, monitor="val_loss")
    early_stopper = EarlyStopping("val_loss", min_delta=0.0008, patience=2, verbose=False)
    callbacks = [pruner, early_stopper]

 

    pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 50,
    "accelerator": "auto",
    "callbacks": callbacks,
    }

    common_model_args = {
        "optimizer_kwargs": {'lr': lr},
        "pl_trainer_kwargs": pl_trainer_kwargs,
        "lr_scheduler_cls": lr_scheduler_cls,
        "lr_scheduler_kwargs": config.lr_scheduler_kwargs,
        "likelihood": None,  # use a likelihood for probabilistic forecasts
        "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
        "force_reset": True,
        "batch_size": batch_size,
        "random_state": 42,

    }

    # build the NBetas model
    model = NBEATSModel(
        generic_architecture=False,
        num_blocks=num_blocks,
        num_layers=num_layers,
        layer_widths=layer_widths,
        loss_fn=torch.nn.MSELoss(),
        input_chunk_length= in_len,
        output_chunk_length = out_len,
        model_name = 'NBEATSModel_84_12_HyperOpt',
        dropout=dropout,
        **common_model_args,

)
    return model

def objective_nbeats(trial):
    series = create_series(dataframe)
    model = create_nbeats_model(trial)
       # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = val
    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    models[model.model_name] = NBEATSModel.load_from_checkpoint("NBEATSModel_84_12_HyperOpt", best=True)

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.historical_forecasts(series=val, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=True, 
                                        forecast_horizon=model.model_params['output_chunk_length']
                                        )
    smapes = rmse(val['dust'], preds['dust'], n_jobs=-1, verbose=True)
    smape_val = np.mean(smapes)

    return smape_val if smape_val != np.nan else float("inf")


def objective_transformer(trial):
    # select input and output chunk lengths
    model = create_transformer_model(trial)
    # when validating during training, we can use a slightly longer validation
    # set which also contains the first input_chunk_length time steps
    model_val_set = val
        # detect if a GPU is available
    if torch.cuda.is_available():
        num_workers = 8
    else:
        num_workers = 0
    # train the model
    model.fit(
        series=train,
        val_series=model_val_set,
        num_loader_workers=num_workers,
    )

    # reload best model over course of training
    models[model.model_name] = TransformerModel.load_from_checkpoint("TransformerModel_84_12_HyperOpt", best=True)

    # Evaluate how good it is on the validation set, using sMAPE
    preds = model.historical_forecasts(series=val, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=True, 
                                        forecast_horizon=model.model_params['output_chunk_length']
                                        )
    # preds = model.predict(series=train, n=VAL_LEN)
    smapes = rmse(val['dust'], preds['dust'], n_jobs=-1, verbose=True)
    smape_val = smapes

    return smape_val if smape_val != np.nan else float("inf")


# %%
def evaluate_model(model, test, forecast_horizon):
    result_accumulator = {}
    print(f'For model {model.model_name}')
    pred_series = model.historical_forecasts(series=test, 
                                        past_covariates=None,
                                        future_covariates=None,
                                        retrain=False,
                                        verbose=False, 
                                        forecast_horizon=forecast_horizon,
                                        overlap_end=True
                                        )
    test_dust = test['dust']
    pred_dust = pred_series['dust']
    result_accumulator[model.model_name] = {
        "mae": mae(test_dust, pred_dust),
        "mse": mse(test_dust, pred_dust),
        "mape": mape(test_dust, pred_dust),
        "rmse": rmse(test_dust, pred_dust)
    }
    return result_accumulator, pred_series

# %%
def train_model(model, val):
    print('model to train', model)
    model.fit(
        series=train,
            val_series=val,
            verbose=False,
            )
    model.save(f'../model_files/{model.model_name}.pt')
    return model


# %%
def logging(model, test, study):
    forecast_horizon = model.model_params['output_chunk_length']
    with mlflow.start_run(run_name=model.model_name):
    # Log the hyperparameters
        mlflow.log_params(model.model_params)

        # Log the loss metric
        eval_results, pred_series = evaluate_model(model, test, forecast_horizon)
        print(eval_results)
        for metric, result in eval_results[model.model_name].items():
            mlflow.log_metric(metric, result)

        # Set a tag that we can use to remind ourselves what this run was for
        mlflow.set_tag("Model_Name", model.model_name)

        # Log the model
        mlflow.log_artifact(f'../model_files/{model.model_name}.pt')

        # fig, ax = plt.subplots(figsize=(20, 5))
        df_to_plot = pd.DataFrame({'Actual': test['dust'].pd_series(), model.model_name: pred_series['dust'].pd_series()})
        fig = df_to_plot.plot(title="Dust Storm Predictions", template="simple_white",
              labels=dict(index="time", value="dust", variable="Legend"))
        # fig.show()
        fig.update_layout(autosize = False, width = 1200, height = 600)
        fig.write_image(f'../plots/{model.model_name}_dust.pdf')

        mlflow.log_artifact(f'../plots/{model.model_name}_dust.pdf')
        joblib.dump(study, f'../optuna/study_{model.model_name}.pkl')
        mlflow.log_artifact(f'../optuna/study_{model.model_name}.pkl')

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize")

import time
curr_time = time.time()

study.optimize(objective_tide, n_trials=20, callbacks=[print_callback])
best_model = create_tide_model(study.best_trial)
best_model = train_model(best_model, val)
logging(best_model, test, study)
tide_time = time.time()
diff = tide_time - curr_time
print(f'Tide took {diff} seconds')
exit()

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize")

import time
curr_time = time.time()

study.optimize(objective_nbeats, n_trials=20, callbacks=[print_callback])
best_model = create_nbeats_model(study.best_trial)
best_model = train_model(best_model, val)
logging(best_model, test, study)

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

n_beats_time = time.time()
diff = n_beats_time - curr_time
print(f'N_beats took {diff} seconds')

study = optuna.create_study(direction="minimize")

study.optimize(objective_tcn, n_trials=20, callbacks=[print_callback])

# %%
best_model = create_tcn_model(study.best_trial)
best_model = train_model(best_model, val)
logging(best_model, test, study)

tcn_time = time.time()
diff = tcn_time - n_beats_time
print(f'TCN took {diff} seconds')

# %%

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize")

study.optimize(objective_block_rnn, n_trials=20, callbacks=[print_callback])

# %%
best_model = create_block_rnn_model(study.best_trial)
best_model = train_model(best_model, val)
logging(best_model, test, study)
# exit()
rnn_time = time.time()
diff = rnn_time - n_beats_time
print(f'RNN took {diff} seconds')

# %%

def print_callback(study, trial):
    print(f"Current value: {trial.value}, Current params: {trial.params}")
    print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

study = optuna.create_study(direction="minimize")

study.optimize(objective_transformer, n_trials=20, callbacks=[print_callback])

# %%
best_model = create_transformer_model(study.best_trial)
best_model = train_model(best_model, val)
logging(best_model, test, study)
 

transformer_time  = time.time()
diff = transformer_time - rnn_time
print(f'Trnasformer took {diff} secinds')


