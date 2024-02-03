
optimizer_kwargs = {
    "lr": 1e-3,
}
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 200,
    "accelerator": "auto",
    "callbacks": [],
}

lr_scheduler_kwargs = {
    "gamma": 0.999,
}


# early stopping (needs to be reset for each model later on)
# this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
early_stopping_args = {
    "monitor": "val_loss",
    "patience": 3,
    "min_delta":0.00008,
    "mode":'min',
}
