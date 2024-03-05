import hydra
import json
import datetime
from pathlib import Path
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
    benchmark_dir = Path(__file__).resolve().parent
    json_file = benchmark_dir.joinpath("results.json")
    if json_file.exists():
        results = json.load(open(json_file, "r"))
    else:
        results = {}

    # Results dictionary
    benchmark_name = benchmark.__class__.__name__
    operator_name = str(operator)
    seed = str(cfg.seed)

    if benchmark_name not in results:
        results[benchmark_name] = {}
    if operator_name not in results[benchmark_name]:
        results[benchmark_name][operator_name] = {}
    if seed not in results[benchmark_name][operator_name]:
        results[benchmark_name][operator_name][seed] = {}

    # Values to save
    results[benchmark_name][operator_name]["timestamp"] = datetime.datetime.now()
    results[benchmark_name][operator_name][seed]["loss/train"] = train_loss.item()
    results[benchmark_name][operator_name][seed]["loss/test"] = test_loss.item()

    # Save results
    json.dump(results, open(json_file, "w"), sort_keys=True, indent=4)


if __name__ == "__main__":
    metric = run()
