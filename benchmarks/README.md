# Benchmarks

## Prerequisites

The benchmarks require optional dependencies:

```bash
pip install -e ".[benchmark]"
```

## Run

Run the `run_all.py` script to run all benchmarks.

```bash
python run_all.py
```

If you only want to evaluate a single benchmark, adopt the `run_single.py` script
and run

```bash
python run_single.py
```

## Visualize

We use MLFlow to log the benchmark runs and you can run

```bash
mlflow ui
```

to visualize the logged experiments in your browser.

In order to visualize the best runs for every benchmark in an HTML table, run

```bash
python process.py
```
