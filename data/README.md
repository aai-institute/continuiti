# Data

## Prerequisites

We use `dvc` to manage the data. You can install the required packages by
installing the benchmarks requirements.

```
pip install -e .[benchmarks]
```

## Downloading the data

The data is stored in a remote storage on GDrive.
To download the data, you can run:

```
cd data
dvc pull <NAME>
```

where `<NAME>` is the name of the data set you want to download,
e.g., `flame` or `navierstokes`, or empty.


## Data sets

### FLAME

`data/flame` contains the dataset from [2023 FLAME AI
Challenge](https://www.kaggle.com/competitions/2023-flame-ai-challenge/data).

### Navier-Stokes

`data/navierstokes` contains a part of the dataset linked in
[neuraloperator/graph-pde](https://github.com/neuraloperator/graph-pde)
(Zongyi Li et al. 2020).
