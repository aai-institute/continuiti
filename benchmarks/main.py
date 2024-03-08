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
    lambda: SineBenchmark(),
]

# Operators
operators = [
    lambda shapes: DeepONet(shapes),
    lambda shapes: DeepONet(
        shapes, trunk_depth=32, branch_depth=32, basis_functions=32
    ),
    lambda shapes: BelNet(shapes),
    lambda shapes: NaiveIntegralKernel(
        kernel=NeuralNetworkKernel(shapes, kernel_width=128, kernel_depth=8)
    ),
]

# Seeds
num_seeds = 10


# === RUN ===


def run_single(seed, benchmark, operator):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    trainer = Trainer(operator, loss_fn=benchmark.metric())
    stats = trainer.fit(benchmark.train_dataset, tol=1e-3)

    loss_test = dataset_loss(benchmark.test_dataset, operator, benchmark.metric())
    stats["loss/test"] = loss_test.item()
    return stats


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

    df = pd.concat(all_results)
    df.to_csv("results.csv")
    df.to_html("results.html")


if __name__ == "__main__":
    run_all()
