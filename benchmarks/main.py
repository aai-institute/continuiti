import hydra
import json
import datetime
import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
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
    stats = trainer.fit(benchmark.train_dataset, tol=cfg.tol)

    # Evaluate on train/test set
    train_loss = dataset_loss(benchmark.train_dataset, operator, benchmark.metric())
    test_loss = dataset_loss(benchmark.test_dataset, operator, benchmark.metric())

    # Results dictionary
    res = OmegaConf.to_container(cfg)
    res["loss/train"] = train_loss.item()
    res["loss/test"] = test_loss.item()
    res["epoch"] = stats["epoch"]

    # Load results from benchmark directory
    benchmark_dir = Path(__file__).resolve().parent
    json_file = benchmark_dir.joinpath("results.json")
    try:
        results = json.load(open(json_file, "r"))
    except (FileNotFoundError, json.decoder.JSONDecodeError):
        results = {}

    # Save new results
    timestamp = str(datetime.datetime.now())
    results[timestamp] = res
    json.dump(results, open(json_file, "w"), sort_keys=True, indent=4)


if __name__ == "__main__":
    metric = run()
