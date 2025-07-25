# OpenMarsML: Machine Learning for Martian Weather Prediction

This repository contains the implementation of deep learning models for time series forecasting of Martian weather conditions using data from the NASA InSight lander and OpenMARS reanalysis dataset.

## Overview

This project implements and evaluates various deep learning architectures for predicting Martian atmospheric variables including:
- Surface pressure (Psurf)
- Air temperature (temp)
- Wind speeds (u_wind, v_wind)
- Dust optical depth (dust)
- Cloud cover (cloud)
- Water vapor (vapour)
- Surface temperature (Tsurf)

## Dataset

The dataset combines observations from NASA's InSight lander with assimilation data from the OpenMARS reanalysis dataset. The data includes:

- **Time series data**: Hourly measurements from the InSight lander
- **Features**: 11 atmospheric variables measured at the InSight landing site
- **Temporal coverage**: Martian days (sols) with corresponding Earth time
- **Data sources**: 
  - InSight observations (pressure, temperature, wind)
  - OpenMARS reanalysis (assimilation data)

### Data Structure
- `data/data_files/train.csv` - Training dataset
- `data/data_files/val.csv` - Validation dataset  
- `data/data_files/test.csv` - Test dataset
- `data/data_files/full_dataset.csv` - Complete dataset

## Models Implemented

### 1. LSTNet (Long- and Short-term Time-series Network)
- **File**: `lstnet_train.py`
- **Architecture**: CNN + GRU with skip connections
- **Features**: 
  - Convolutional layers for local pattern extraction
  - GRU layers for temporal dependencies
  - Skip connections for long-term patterns
  - Highway connections for autoregressive components

### 2. Darts Library Models
- **File**: `notebooks/train_tune.py`
- **Models**: 
  - BlockRNNModel (LSTM/GRU)
  - TCNModel (Temporal Convolutional Network)
  - TransformerModel
  - TiDEModel
  - NBEATSModel

## Hyperparameter Optimization

The project uses Optuna for hyperparameter tuning with the following search spaces:

### LSTNet Parameters
- Window size: 84 time steps
- Horizon: 12 time steps
- Hidden CNN units: 30
- Hidden RNN units: 30
- Learning rate: 0.001
- Batch size: 128
- Dropout: 0.2

### Darts Models Parameters
- Input chunk length: 84
- Output chunk length: 12
- Batch size: [32, 96, 128, 256]
- Hidden size: [64, 128, 256]
- Learning rate: [5e-5, 1e-3]
- Dropout: [0.05, 0.25]

## Project Structure

```
OpenMarsML/
├── data/                          # Dataset files
│   ├── data_files/               # Train/val/test splits
│   ├── predicted_data/           # Model predictions
│   └── data_import_scripts/      # Data preprocessing scripts
├── notebooks/                     # Jupyter notebooks and training scripts
│   ├── train_tune.py            # Main training script with Optuna
│   ├── darts_all_models_*.ipynb # Model comparison notebooks
│   └── mlruns/                  # MLflow experiment tracking
├── lstnet_train.py              # LSTNet training script
├── lstnet_inference.py          # LSTNet inference script
├── lstnet_predict.py            # LSTNet prediction script
├── dashboard.py                 # Interactive visualization dashboard
├── utils/                       # Utility functions
│   └── config/                  # Configuration files
├── plots/                       # Generated plots and visualizations
├── metrics/                     # Evaluation metrics
└── requirements.txt             # Python dependencies
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd OpenMarsML
```

2. Create and activate conda environment:
```bash
conda create --name OpenMarsML --file requirements.txt
conda activate OpenMarsML
```

3. Install additional dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training LSTNet Model

```bash
python lstnet_train.py
```

The script will:
- Load the training, validation, and test datasets
- Train the LSTNet model with specified hyperparameters
- Save the best model based on validation loss
- Evaluate on test set and print metrics

### Training Darts Models with Hyperparameter Tuning

```bash
cd notebooks
python train_tune.py
```

This script:
- Implements Optuna-based hyperparameter optimization
- Trains multiple Darts models (BlockRNN, TCN, Transformer, TiDE, N-BEATS)
- Uses MLflow for experiment tracking
- Saves best models and metrics

### Inference

```bash
python lstnet_inference.py
```

### Interactive Dashboard

```bash
python dashboard.py
```

Access the dashboard at `http://localhost:8050` to visualize:
- Actual vs predicted values
- Model comparisons
- Different atmospheric variables

## Evaluation Metrics

The models are evaluated using:
- **MAE** (Mean Absolute Error)
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)
- **MAPE** (Mean Absolute Percentage Error)

### Sample Results (BlockRNNModel_84_12)
- Surface Pressure: RMSE = 0.078, MAPE = 23.24%
- Temperature: RMSE = 0.170, MAPE = 20.32%
- Dust: RMSE = 0.512, MAPE = 77.92%

## Key Findings

1. **Model Performance**: BlockRNN and TCN models show competitive performance across all variables
2. **Variable Difficulty**: Dust prediction is most challenging (high MAPE), while pressure prediction is most accurate
3. **Temporal Patterns**: Models capture both short-term (hourly) and long-term (seasonal) patterns
4. **Hyperparameter Sensitivity**: Window size and horizon significantly impact prediction accuracy

## Research Contributions

1. **First application** of deep learning time series models to Martian weather prediction
2. **Comprehensive evaluation** of multiple architectures on real Martian data
3. **Hyperparameter optimization** using Optuna for optimal model performance
4. **Interactive visualization** tools for model analysis and comparison

## Dependencies

- **Deep Learning**: PyTorch, PyTorch Lightning
- **Time Series**: Darts
- **Optimization**: Optuna
- **Visualization**: Plotly, Dash, Matplotlib
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Experiment Tracking**: MLflow

## Citation

If you use this code in your research, please cite:

```bibtex
@article{openmarsml2024,
  title={Deep Learning for Martian Weather Prediction: A Comparative Study},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2024}
}
```

## Contact

For questions or issues, please contact: [Your Email]

## License

[Specify your license here] 