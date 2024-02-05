# Project Name

## Description

Extra simulations for the paper "Interpretable Deep Image Classification using Rationally Inattentive Utility Maximization" done to validate the findings on newer datasets and architectures.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Installation

Instructions on how to install and set up the project.
Required dependencies:
- `torch`
- `numpy`
- `matplotlib`
- `pandas`
- `seaborn`
- `scipy`
- `scikit-learn`
- `huggingface/transformers`
- `tqdm`
- `huggingface/datasets`


## Usage

Directory structure:
- 'cm' -> Confusion matrices for both the experiments
- 'utils' -> Reconstructed utilities for the experiments
- 'plots' -> Plots for the experiments

Files:
- 'generate_confusion_matrix_epoch.py' -> Generates confusion matrices for the experiment with epochs as decision problem
- 'generate_confusion_matrix_variance.py' -> Generates confusion matrices for the experiment with variance as decision problem
- 'optimize_maxmargin.py' -> Reconstructs the max-margin utility for the experiments
- 'optimize_sparse.py' -> Reconstructs the sparse utility for the experiments
- 'generate_plots.py' -> Generates plots for the experiments
##
## License

### MIT License

