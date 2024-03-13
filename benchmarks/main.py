import random
import numpy as np
import pandas as pd
import torch
from continuity.benchmarks.sine import SineBenchmark
from continuity.operators import DeepONet, BelNet
from continuity.operators.integralkernel import NaiveIntegralKernel, NeuralNetworkKernel
from continuity.trainer import Trainer
from continuity.data.utility import dataset_loss


# Benchmarks
benchmarks = [
    lambda: SineBenchmark(32),
    lambda: SineBenchmark(128),
]

# Operators
operators = [
    lambda s: DeepONet(s),
    lambda s: DeepONet(s, trunk_depth=32, branch_depth=32, basis_functions=32),
    lambda s: BelNet(s, D_1=16, D_2=16),
    lambda s: NaiveIntegralKernel(
        kernel=NeuralNetworkKernel(s, kernel_width=32, kernel_depth=3)
    ),
]

# Seeds
num_seeds = 2

# Training parameters
lr = 3e-5
tol = 1e-4
epochs = 3


# === RUN ===
def run_single(seed, benchmark, operator):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(operator, lr=lr, loss_fn=benchmark.metric())

    stats = trainer.fit(benchmark.train_dataset, tol=tol, epochs=epochs)

    stats["loss/test"] = dataset_loss(
        benchmark.test_dataset, operator, benchmark.metric()
    )
    return stats


def write_results(all_results):
    df = pd.concat(all_results)
    df.to_csv("results.csv")

    html_string = """
<html>
<head><title>Continuity Benchmark</title></head>
<link rel="stylesheet" type="text/css" href="style.css"/>
<body>
    {table}
</body>
</html>
    """
    with open("results.html", "w") as f:
        f.write(html_string.format(table=df.to_html()))


def run_all():
    all_results = {}

    for benchmark in benchmarks:
        results = []

        for operator in operators:
            for seed in range(num_seeds):
                bm = benchmark()
                op = operator(bm.dataset.shapes)

                print(f"=== {bm} {op} [{seed}]")
                stats = run_single(seed, bm, op)

                results.append({"Operator": str(op), **stats})
                bm_key = str(bm)

        df = pd.DataFrame(results)

        # Compute mean loss over seeds
        df = df.groupby(["Operator"]).agg(
            {
                "loss/train": ["mean", "std"],
                "loss/test": ["mean", "std"],
                "epoch": ["mean", "std"],
            }
        )

        all_results[bm_key] = df

    write_results(all_results)


if __name__ == "__main__":
    run_all()
