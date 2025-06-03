# SBCube MetaML Project

This repository contains the implementation of the SLT experiments for SBCube MetaML project.

## Table of Contents

- [Overview](#overview)
- [Requirements](#requirements)
- [Usage](#usage)
- [Arguments and Hyperparameters](#arguments-and-hyperparameters)

---

## Overview

The following main components are included:
- `main.py`: Entry point for training and evaluation
- `models/`: Model and layer definitions
- `utils/`: Utility functions (including dataset loading, seed setting, etc.)

---

## Requirements

Install the experiment environment using:

```bash
conda create -n metaml_slt python=3.x
conda activate metaml_slt
conda install --yes --file requirements.txt
```

**Note:**
Python version: `>=3.x`

---

## Usage

Run the experiment with all relevant hyperparameters via the command line:

```bash
python main.py               \
--dataset          mnist     \
--model            lenet5    \
--epochs           100       \
--batch_size       128       \
--lr               0.01      \
--seed             0         \
--save_dir         ./results \
--slt_sparsity     0.5       \
--n_dropout_layers 0         \
--p_dropout        0.0       \
--n_samples        1         \
--width_scale      1
```

---

## Arguments and Hyperparameters

You can modify the behavior of the script using the following arguments:

### General Training Settings

| Argument        | Type     | Default     | Description                                                                 |
|----------------|----------|-------------|-----------------------------------------------------------------------------|
| `--dataset`     | `str`    | `mnist`     | Dataset to use (`mnist`)                                        |
| `--model`       | `str`    | `lenet5`    | Model architecture (`lenet5`)                                              |
| `--epochs`      | `int`    | `100`       | Number of training epochs                                                  |
| `--batch_size`  | `int`    | `128`       | Batch size for training and evaluation                                     |
| `--lr`          | `float`  | `0.01`      | Initial learning rate                                                      |
| `--seed`        | `int`    | `0`         | Random seed for reproducibility                                            |
| `--save_dir`    | `str`    | `./results` | Directory to save logs and model checkpoints                               |

The saved results (in CSV format) are structured as follows:
`epoch, val_acc, test_acc, model_size, aPE, ECE, FLOPs, MACs`
For example:
```
epoch, val_acc, test_acc, model_size, aPE, ECE, FLOPs, MACs
0,     10.0,    10.0,     1.0MB,      0.1,  0.1,   100,   100
1,     11.2,    10.9,     1.0MB,      0.1,  0.1,   100,   100
2,     13.4,    12.1,     1.0MB,      0.1,  0.1,   100,   100
...
```

---

### MetaML and SLT Settings

| Argument           | Type     | Default | Description                                                                 |
|--------------------|----------|---------|-----------------------------------------------------------------------------|
| `--slt_sparsity`   | `float`  | `0.5`   | Global sparsity for the Strong Lottery Ticket (NOTE: We use global EP method for finding SLTs)                |
| `--n_dropout_layers` | `int`  | `0`     | Number of dropout layers (NOTE: Dropout layers are added in order from the last layer side)        |
| `--p_dropout`      | `float`  | `0.0`   | Dropout probability for each dropout layer                                 |
| `--n_samples`      | `int`    | `1`     | Number of Monte Carlo samples used for evaluation                          |
| `--width_scale`    | `int`    | `1`     | Width multiplier for model layers; scales hidden units or channels by this factor |
