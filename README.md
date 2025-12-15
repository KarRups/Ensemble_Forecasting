# Ensemble Forecasting for Hydrological Prediction

Code accompanying the paper on temporally consistent ensemble forecasting for river discharge prediction using machine learning.

## Overview

This repository implements methods for generating **temporally consistent** ensemble forecasts of river discharge at multiple lead times (up to 10 days). Traditional ML approaches for multi-day forecasting suffer from two key problems:

1. **Deterministic models** are mean-seeking - they predict the most likely flow for each individual day, resulting in physically implausible trajectories
2. **Probabilistic models** provide no insight into how uncertainties connect across lead times

We introduce two novel approaches that enforce temporal consistency:
- **Conditional-LSTM**: Treats predictions at previous lead times as ground truth during training
- **Seeded-LSTM**: Uses ensemble generation techniques to create physically consistent multi-day trajectories

These methods successfully predict temporal properties of 10-day hydrographs (e.g., accumulated flow, days over threshold) and allow efficient generation of arbitrary ensemble sizes.

## Key Features

- Multi-basin river discharge forecasting (2,450+ catchments)
- 10-day forecast horizon
- Temporally consistent ensemble generation
- Comparison with standard ML forecasting approaches
- Evaluation of temporal properties (not just point-wise accuracy)

## Installation

```bash
git clone https://github.com/KarRups/Ensemble_Forecasting.git
cd Ensemble_Forecasting
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- PyTorch
- xarray
- zarr
- scikit-learn
- pandas, numpy, matplotlib
- See `requirements.txt` for full dependencies

## Project Structure

```
Ensemble_Forecasting/
├── ML_Functions/          # Core model implementations
│   ├── ML_functions.py    # Data loading and preprocessing
│   ├── ML_Models.py       # LSTM architectures (Conditional, Seeded)
│   ├── ML_Losses.py       # Custom loss functions
│   ├── ML_Processing.py   # Post-processing utilities
│   └── ML_Plots.py        # Visualization tools
├── ML_Training/           # Training scripts and configurations
│   ├── GPU_Training.py    # Main training script
│   ├── Making_Dataloaders.py  # Dataset preparation
│   └── Evaluation_Results.py  # Model evaluation
├── Catchment_Averaging/   # Basin selection and preprocessing
│   ├── Training_Models.ipynb
│   └── ML_basin_splits.pkl    # Train/val/test splits
├── notebooks/             # Analysis notebooks
│   ├── Analysing_Models.ipynb
│   └── Choosing_Catchment_Sample.ipynb
└── README.md
```

## Data Requirements

This project uses the following dataset:
**CARAVAN-Multimet dataset** - Multi-basin discharge observations and static attributes

### Data Setup

Due to data size, meteorological and discharge data are not included in this repository. You will need to:

2. Download CARAVAN basin data: [link to CARAVAN]
3. Set up your data directory structure (see `DATA_SETUP.md`)
4. Update paths in `config.py` to point to your data location
## Quick Start

### Training a Model

```python
# Example: Train Conditional-LSTM model
python ML_Training/GPU_Training.py --model conditional --epochs 50 --batch_size 256
```

### Generating Ensemble Forecasts

```python
from ML_Functions.ML_Models import ConditionalLSTM
from ML_Functions.ML_functions import load_data

# Load trained model
model = ConditionalLSTM.load('Models/trained_model.pth')

# Generate 50-member ensemble
ensemble = model.generate_ensemble(input_data, n_members=50)
```

### Example Analysis

See `notebooks/Analysing_Models.ipynb` for a complete workflow including:
- Loading and preprocessing data
- Training models
- Generating ensembles
- Evaluating temporal consistency
- Comparing with baseline methods

## Model Architectures

### Conditional-LSTM
Explicitly trains the model to treat previous lead time predictions as ground truth, ensuring temporal consistency by conditioning each forecast step on the previous prediction.

### Seeded-LSTM
Generates ensembles by perturbing initial conditions and propagating uncertainty through the forecast horizon, creating physically plausible trajectory spread.

## Evaluation Metrics

Beyond traditional point-wise metrics (NSE, KGE), we evaluate:
- **CRPS** (Continuous Ranked Probability Score) for probabilistic skill
- **Accumulated flow accuracy** over 10-day windows
- **Days-over-threshold** prediction skill
- **Trajectory plausibility** metrics

## Citation

If you use this code in your research, please cite:

```
```

## Contributing

This code is provided for research reproducibility. For questions or issues, please open a GitHub issue or contact [your email].

## License

MIT License - see LICENSE file for details

## Acknowledgments

- CARAVAN dataset: Kratzert et al. (2023)
- ERA5-Land: Copernicus Climate Change Service
- ECMWF HRES forecasts

## Authors


---

**Note for Users:** This is research code developed for a specific study. While we've made efforts to make it usable, some paths and configurations may need adjustment for your environment. Please see `DATA_SETUP.md` for detailed setup instructions and feel free to open issues for questions.
