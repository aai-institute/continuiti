import torch
from continuity.benchmarks.sine import SineBenchmark
from continuity.callbacks import OptunaCallback
from continuity.data import split, dataset_loss
from continuity.operators import DeepONet
import optuna

# Set random seed
torch.manual_seed(0)


def test_optuna():
    def objective(trial):
        trunk_width = trial.suggest_int("trunk_width", 4, 16)
        trunk_depth = trial.suggest_int("trunk_depth", 4, 16)
        num_epochs = trial.suggest_int("num_epochs", 1, 10)
        lr = trial.suggest_float("lr", 1e-4, 1e-3)

        # Data set
        benchmark = SineBenchmark()

        # Train/val split
        train_dataset, val_dataset = split(benchmark.train_dataset, 0.9)

        # Operator
        operator = DeepONet(
            benchmark.dataset.shape,
            trunk_width=trunk_width,
            trunk_depth=trunk_depth,
        )

        # Optimizer
        optimizer = torch.optim.Adam(operator.parameters(), lr=lr)

        operator.compile(optimizer, verbose=False)
        operator.fit(
            train_dataset, epochs=num_epochs, callbacks=[OptunaCallback(trial)]
        )

        loss_val = dataset_loss(val_dataset, operator, benchmark.metric())
        print(f"loss/val: {loss_val:.4e}")

        return loss_val

    # Run hyperparameter optimization
    name = "test_optuna"
    study = optuna.create_study(
        direction="minimize",
        study_name=name,
        storage=f"sqlite:///{name}.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=10)


if __name__ == "__main__":
    test_optuna()
