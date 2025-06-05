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
- `modells.py`: Model and layer definitions
- `utils/`: Utility functions (including dataset loading, seed setting, etc.)

---

## Requirements

Install the experiment environment using:

```bash
conda env create -f requirements.yml
conda activate metaml_slt
```
<!-- 
**Note:**
Python version: `>=3.x` -->

---

## Usage

Run the experiment with all relevant hyperparameters via the command line:

### Original SLT
```bash
python main.py       \
--dataset mnist      \
--model lenet        \
--slt                \
--pruning_rate 0.50  \
--scaling_rate 1.00  \
--dropout_rate 0.05  \
--num_bayes_layers 1 \
```

### Partial Frozen SLT
```bash
python main.py       \
--dataset mnist      \
--model lenet        \
--partial_frozen_slt \
--pruning_rate 0.50  \
--scaling_rate 1.00  \
--dropout_rate 0.05  \
--num_bayes_layers 1 \
```

---

## Arguments and Hyperparameters

You can modify the behavior of the script using the following arguments:

### General Training Settings

| Argument        | Type     | Default     | Description                                                                 |
|----------------|----------|-------------|-----------------------------------------------------------------------------|
| `--dataset`     | `str`    | `mnist`     | Dataset to use (`mnist`)                                        |
| `--model`       | `str`    | `lenet`    | Model architecture (`lenet5`)                                              |
| `--epochs`      | `int`    | `100`       | Number of training epochs                                                  |
| `--batch_size`  | `int`    | `128`       | Batch size for training and evaluation                                     |
| `--lr`          | `float`  | `0.1`      | Initial learning rate                                                      |
| `--seed`        | `int`    | `110`         | Random seed for reproducibility                                            |
<!-- | `--save_dir`    | `str`    | `./results` | Directory to save logs and model checkpoints                               | -->

The saved results (in log format) are structured as follows:
`Epoch, train_acc, val_acc, test_acc, ECE, aPE, Accuracy, FLOPS` \
For example:
```
----------------------------------------
Training Results:
Epoch, train_acc, val_acc, test_acc
----------------------------------------
1	10.96	9.69	9.66
2	11.55	9.88	10.05
3	12.13	11.37	10.74
...
100	97.58	98.91	98.82
----------------------------------------
Final Results
ECE(%), aPE(nats), Accuracy(%), FLOPS(10^6)
----------------------------------------
8.16	1.3642	98.70	5.78
----------------------------------------
```

---

### MetaML and SLT Settings

| Argument           | Type     | Default | Description                                                                 |
|--------------------|----------|---------|-----------------------------------------------------------------------------|
| `--pruning_rate`   | `float`  | `0.5`   | Global sparsity for the Strong Lottery Ticket (NOTE: We use global EP method for finding SLTs) |
| `--scaling_rate`   | `float`  | `1.0`   | Width multiplier for hidden units or channels by this factor |
| `--dropout_rate`      | `float`  | `0.05`   | Dropout probability for each dropout layer |
| `--num_bayes_layers` | `int`  | `1`     | Number of dropout layers (NOTE: Dropout layers are added in order from the last layer side) |
| `--mc_samples`      | `int`    | `10`     | Number of Monte Carlo samples used for evaluation |

