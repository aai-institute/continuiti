# Benchmarks

## Run

If you want to evaluate a single benchmark, adopt the `run_single.py` script
and (within the `benchmarks` directory) run

```bash
python run_single.py
```

To run a specific benchmark, e.g., the sine benchmarks, run

```bash
python sine/run_all.py
```

## MLFlow

We use MLFlow to log the benchmark runs and you can run

```bash
mlflow ui
```

to visualize the logged experiments in your browser.

## Documentation

In order to process the data of the best runs for the benchmark page in
the documentation, run

```bash
python results/process.py
```
