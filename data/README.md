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
dvc pull
```
