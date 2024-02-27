# Benchmarks

## Pre-requisites

To run the benchmarks, you need to install additional dependencies:

```bash
pip install -e ".[benchmarks]"
```

## Running benchmarks

In the `benchmarks` directory, you can find a `main.py` script that runs the benchmarks.

You can run all benchmarks with:

```bash
python main.py
```

Alternatively, you can run a specific benchmark (e.g., fixing the operator to `BelNet`) with:

```bash
python main.py -m operator=belnet
```
