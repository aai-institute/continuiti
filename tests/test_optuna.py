import pytest
import torch
from continuity.benchmarks.sine import SineBenchmark
from continuity.trainer import Trainer
from continuity.trainer.callbacks import OptunaCallback, PrintTrainingLoss
from continuity.data import split, dataset_loss
from continuity.operators import DeepONet
import optuna


@pytest.mark.slow
def test_optuna():
    def objective(trial):
        trunk_width = trial.suggest_int("trunk_width", 32, 64)
        trunk_depth = trial.suggest_int("trunk_depth", 4, 8)
        lr = trial.suggest_float("lr", 1e-4, 1e-3)

        # Data set
        benchmark = SineBenchmark()

        # Train/val split
        train_dataset, val_dataset = split(benchmark.train_dataset, 0.9)

        # Operator
        operator = DeepONet(
            benchmark.dataset.shapes,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
        )

        # Optimizer
        optimizer = torch.optim.Adam(operator.parameters(), lr=lr)

        trainer = Trainer(operator, optimizer)
        trainer.fit(
            train_dataset,
            tol=1e-2,
            callbacks=[PrintTrainingLoss(), OptunaCallback(trial)],
        )

        loss_val = dataset_loss(val_dataset, operator, benchmark.metric())
        print(f"loss/val: {loss_val:.4e}")

        return loss_val

    # Run hyperparameter optimization
    name = "test_optuna"
    study = optuna.create_study(
        direction="minimize",
        study_name=name,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=3)


if __name__ == "__main__":
    test_optuna()
