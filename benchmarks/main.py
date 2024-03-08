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
    results = defaultdict(lambda: defaultdict(dict))
    if json_file.is_file():
        old_results = json.load(open(json_file, "r"))
        results.update(old_results)

    # Evaluate on train/test set
    for mt in benchmark.metrics:
        train_result = mt.calculate(operator, benchmark.train_dataset)
        test_result = mt.calculate(operator, benchmark.test_dataset)
        result = {"train": train_result, "test": test_result}
        results[benchmark_name][operator_name][str(mt)] = result

    # Save results
    json.dump(results, open(json_file, "w"), sort_keys=True, indent=4)


if __name__ == "__main__":
    run()
