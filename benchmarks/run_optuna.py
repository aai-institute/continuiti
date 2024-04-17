import optuna
import torch
from functools import partial
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import SineRegular
from continuiti.operators import (
    FourierNeuralOperator,
)

# FFT not available on MPS
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    def objective(trial):
        seed = trial.suggest_int("seed", 0, 100)
        width = trial.suggest_int("width", 1, 4)
        depth = trial.suggest_int("depth", 1, 4)

        config = RunConfig(
            SineRegular,
            partial(FourierNeuralOperator, width=width, depth=depth),
            max_epochs=100,
            seed=seed,
            device=device,
        )

        test_loss = BenchmarkRunner.run(config, trial.params)
        return test_loss

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
