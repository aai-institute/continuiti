# Benchmarks

## Pre-requisites

To run the benchmarks, you need to install additional dependencies:

```bash
pip install -e ".[benchmarks]"
```

## Running benchmarks

In the `benchmarks` directory, you can find a `main.py` script that runs the benchmarks.

You can run a specific configuration (e.g., fixing the operator to a small `DeepONet`) with:

```bash
python main.py -m +operator=deeponet_small
```
