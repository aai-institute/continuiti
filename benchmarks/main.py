import hydra
import random
import numpy as np
import torch
import mlflow
from omegaconf import DictConfig
from continuity.data.utility import dataset_loss
from continuity.trainer.callbacks import PrintTrainingLoss, MLFlowLogger


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
    mlflow.start_run()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    mlflow.log_param("seed", cfg.seed)

    # Log benchmark parameters
    cfg_benchmark = dict(cfg.benchmark)
    cfg_benchmark["benchmark"] = cfg_benchmark.pop("_target_")
    mlflow.log_params(cfg_benchmark)

    # Log operator parameters
    cfg_op = dict(cfg.operator)
    cfg_op["operator"] = cfg_op.pop("_target_")
    mlflow.log_params(cfg_op)

    # Log trainer parameters
    cfg_trainer = dict(cfg.trainer)
    cfg_trainer["trainer"] = cfg_trainer.pop("_target_")
    mlflow.log_params(cfg_trainer)

    benchmark = hydra.utils.instantiate(cfg.benchmark)
    operator = hydra.utils.instantiate(cfg.operator, shapes=benchmark.dataset.shapes)
    optimizer = hydra.utils.instantiate(
        cfg.trainer.optimizer, params=operator.parameters()
    )
    trainer = hydra.utils.instantiate(
        cfg.trainer, operator=operator, optimizer=optimizer, loss_fn=benchmark.metric()
    )

    # Train
    callbacks = [PrintTrainingLoss(), MLFlowLogger()]
    stats = trainer.fit(benchmark.train_dataset, tol=cfg.tol, callbacks=callbacks)

    # Save model
    torch.save(operator.state_dict(), "model.pth")
    mlflow.log_artifact("model.pth")

    # Evaluate on test set
    test_loss = dataset_loss(benchmark.test_dataset, operator, benchmark.metric())
    mlflow.log_metric("loss/test", test_loss.item())

    mlflow.log_metric("epoch", stats["epoch"])
    mlflow.end_run()


if __name__ == "__main__":
    metric = run()
