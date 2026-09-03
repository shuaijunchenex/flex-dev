# Data Distribution Configurations

This directory contains YAML configuration files for defining data distributions across clients in Federated Learning (FL) experiments.

## Naming Convention

Files in this directory follow a standardized naming pattern to identify the dataset, distribution characteristics, and client counts:

`<dataset_name>_noniid_<distribution_type>_<client_number>.yaml`

- **`<dataset_name>`**: The identifier for the dataset (e.g., `mnist`, `cifar10`, `mrpc`, `sst2`).
- **`<distribution_type>`**:
  - `extreme`: Each client holds only one specific label's samples (maximum heterogeneity).
  - `balance`: Samples for each label are distributed equally among participating clients.
- **`<client_number>`**: The total number of clients participating in the simulation.

## Dataset Sample Statistics

Reference counts for the processed datasets:

- **MRPC**: [0: 1323, 1: 2753]
- **AG News**: [1: 30000, 2: 30000, 3: 30000, 4: 30000]
- **SST2**: [0: 29780, 1: 37569]
- **QQP**: [0: 229468, 1: 134378]
- **IMDB**: [neg: 12500, pos: 12500]
- **MNLI**: [0: 130899, 1: 130900, 2: 130903]
- **CoLA**: [0: 2528, 1: 6022]
- **MNIST**: [0: 5923, 1: 6742, 2: 5958, 3: 6131, 4: 5842, 5: 5421, 6: 5918, 7: 6265, 8: 5851, 9: 5949]

## Distribution Overview

### NLP Datasets (GLUE & AG News)

| Dataset | Type | Clients | Description |
| :--- | :--- | :--- | :--- |
| **AG News** | extreme | 4 | 4 clients, each holding all samples of one class (World, Sports, Business, Sci/Tech). |
| **AG News** | balance | 10 | 10 clients, samples of all 4 labels distributed equally. |
| **MRPC** | extreme | 2 | 2 clients, split by binary labels (not-equivalent vs equivalent). |
| **MRPC** | balance | 10 | 10 clients, binary labels shared equally. |
| **SST2** | extreme | 2 | 2 clients, split by sentiment (binary). |
| **SST2** | balance | 10 | 10 clients, sentiment labels shared equally. |
| **QQP** | extreme | 2 | 2 clients, split by binary labels (duplicate vs not-duplicate). |
| **QQP** | balance | 10 | 10 clients, labels shared equally. |
| **IMDB** | extreme | 2 | 2 clients, split by positive/negative sentiment. |
| **IMDB** | balance | 10 | 10 clients, sentiment labels shared equally. |
| **MNLI** | extreme | 3 | 3 clients, split by entailment, neutral, and contradiction. |
| **MNLI** | balance | 10 | 10 clients, 3 labels shared equally. |
| **CoLA** | balance | 2 | 2 clients, linguistic acceptability shared equally. |

### CV Datasets

| Dataset | Type | Clients | Description |
| :--- | :--- | :--- | :--- |
| **MNIST** | extreme | 10 | 10 clients, each holding all samples of one digit (0-9). |
| **MNIST** | balance | 10 | 10 clients, all digits present on every client; all 60,000 samples retained and each client has 6,000 samples. |
| **CIFAR10** | extreme | 10 | 10 clients, each holding all samples of one object category. |
| **FMNIST** | extreme | 10 | 10 clients, each holding all samples of one clothing type. |
| **KMNIST** | extreme | 10 | 10 clients, each holding all samples of one Hiragana character. |

## Usage Reference

To use these distributions in your test configuration:

```yaml
yaml_folder_data_distribution_files:
  - sst2_noniid_extreme_2.yaml: sst2_extreme

yaml_combination:
  client_yaml:
    - sst2_extreme
```
