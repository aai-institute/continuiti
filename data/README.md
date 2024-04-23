# Data

## Prerequisites

We use `dvc` to manage the data. Install the required packages by:

```
pip install dvc dvc-gdrive
```

## Downloading the data

The data is stored in a remote storage on GDrive, so you need to authenticate
with the GDrive.

To download, e.g., the `navierstokes` data, run within the `data` directory:

```
dvc pull navierstokes
```

To download all data sets, run:

```
dvc pull
```

Below, you can find a list of available data sets.

## Data sets

### FLAME

`data/flame` contains the dataset from [2023 FLAME AI
Challenge](https://www.kaggle.com/competitions/2023-flame-ai-challenge/data).

### Navier-Stokes

`data/navierstokes` contains a part of the dataset linked in
[neuraloperator/graph-pde](https://github.com/neuraloperator/graph-pde)
(Zongyi Li et al. 2020).
