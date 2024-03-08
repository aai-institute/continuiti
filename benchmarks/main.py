import pathlib
import hydra
import json
from omegaconf import DictConfig
from collections import defaultdict


@hydra.main(version_base="1.3", config_path="configs", config_name="config.yaml")
def run(cfg: DictConfig) -> None:
    """Main entry point for running benchmarks.

    Example usage:
        ```
        python run.py
        ```

    Args:
        cfg: Configuration.
    """
    if cfg.get("seed"):
        import random
        import numpy as np
        import torch

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

    benchmark = hydra.utils.instantiate(cfg.benchmark)
    operator = hydra.utils.instantiate(
        cfg.operator, shapes=benchmark.train_dataset.shapes
    )
    optimizer = hydra.utils.instantiate(
        cfg.trainer.optimizer, params=operator.parameters()
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, operator=operator, optimizer=optimizer
    )

    # Train
    trainer.fit(benchmark.train_dataset)

    # ----- RESULTS -----
    # Results dictionary
    benchmark_name = benchmark.__class__.__name__
    operator_name = str(operator)
    benchmark_dir = pathlib.Path(__file__).parent
    json_file = benchmark_dir.joinpath("results.json")

    # Load results from benchmark directory
    if json_file.is_file():
        old_results = json.load(open(json_file, "r"))

    # Evaluate on train/test set
    results = defaultdict(lambda: defaultdict(dict))
    for mt in benchmark.metrics:
        train_result = mt.calculate(operator, benchmark.train_dataset)
        test_result = mt.calculate(operator, benchmark.test_dataset)
        result = {"train": train_result, "test": test_result}
        results[benchmark_name][operator_name][str(mt)] = result

    # Update and save results
    old_results = load_benchmark_data(json_file)
    results = update_benchmark_data(old_results, results)
    save_benchmark_data(json_file, results)


def load_benchmark_data(file_path: pathlib.Path) -> dict:
    """Load benchmark data from a JSON file."""
    if file_path.exists():
        with open(file_path, "r") as file:
            return json.load(file)
    else:
        return {}  # Return an empty dict if file doesn't exist


def recursive_dict_merge(dict_a: dict, dict_b: dict) -> dict:
    """Recursively merges two dictionaries.

    Recursively merges two dictionaries. If one value in the dictionary is a leaf of the tree it combines both values
    into a list with the value of dict_a first.

    Args:
        dict_a: Dictionary to merge with dict_b.
        dict_b: Dictionary to merge with dict_a.

    Returns:
        Merged dictionary with values from both dictionaries.
    """
    for key, value in dict_b.items():
        if key not in dict_a:
            dict_a[key] = value
            continue

        # key in dict
        if not isinstance(value, dict) or not isinstance(dict_a[key], dict):
            # one dict can not be traversed recursively anymore
            dict_a[key] = [dict_a[key], value]
            continue

        # value in both dicts is still a dict
        dict_a[key] = recursive_dict_merge(dict_a[key], value)

    return dict_a


def update_benchmark_data(existing_data: dict, new_data: dict) -> dict:
    """Update the existing benchmark data with new data."""
    return recursive_dict_merge(existing_data, new_data)


def save_benchmark_data(file_path: pathlib.Path, data: dict):
    """Save the updated benchmark data to a JSON file."""
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4, sort_keys=True)


if __name__ == "__main__":
    run()
