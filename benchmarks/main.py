import os
import hydra
import json
from omegaconf import DictConfig
from continuity.data.utility import dataset_loss


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
    operator = hydra.utils.instantiate(cfg.operator, shapes=benchmark.dataset.shapes)
    optimizer = hydra.utils.instantiate(
        cfg.trainer.optimizer, params=operator.parameters()
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, operator=operator, optimizer=optimizer, loss_fn=benchmark.metric()
    )

    # Train
    trainer.fit(benchmark.train_dataset)

    # Evaluate on train/test set
    train_loss = dataset_loss(benchmark.train_dataset, operator, benchmark.metric())
    test_loss = dataset_loss(benchmark.test_dataset, operator, benchmark.metric())

    # Load results from benchmark directory
    benchmark_dir = os.path.dirname(os.path.realpath(__file__))
    json_file = os.path.join(benchmark_dir, "results.json")
    if os.path.exists(json_file):
        results = json.load(open(json_file, "r"))
    else:
        results = {}

    # Results dictionary
    benchmark_name = benchmark.__class__.__name__
    operator_name = str(operator)

    if benchmark_name not in results:
        results[benchmark_name] = {}
    if operator_name not in results[benchmark_name]:
        results[benchmark_name][operator_name] = {}

    # Values to save
    results[benchmark_name][operator_name]["loss/train"] = train_loss.item()
    results[benchmark_name][operator_name]["loss/test"] = test_loss.item()

    # Save results
    json.dump(results, open(json_file, "w"), sort_keys=True, indent=4)


if __name__ == "__main__":
    metric = run()
