import mlflow
import json
import pandas as pd


def process():
    client = mlflow.MlflowClient()
    experiments = client.search_experiments()
    experiment_ids = [e.experiment_id for e in experiments]
    runs = client.search_runs(experiment_ids)

    all_runs_dicts = []
    for run in runs:
        all_runs_dicts.append(
            {
                "benchmark": run.data.tags["benchmark"],
                "operator": run.data.tags["operator"],
                "num_params": run.data.metrics["num_params"],
                "loss/train": run.data.metrics["loss/train"],
                "loss/test": run.data.metrics["loss/test"],
                "params": run.data.params,
            }
        )

    df = pd.concat([pd.DataFrame([r]) for r in all_runs_dicts])

    filter_params = ["tol", "seed", "max_epochs", "lr", "batch_size"]

    data = {}

    # Load existing data
    try:
        with open("results/data.json", "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        pass

    benchmarks = sorted(df["benchmark"].unique())
    for bm in benchmarks:
        if bm not in data:
            data[bm] = []

        df_benchmark = df[df["benchmark"] == bm]
        operators = sorted(df_benchmark["operator"].unique())

        for operator in operators:
            operator_data = df_benchmark[df_benchmark["operator"] == operator]

            # Get the best operator run for this benchmark
            best = operator_data.sort_values("loss/test").iloc[0].to_dict()

            # Check if the operator already exists with a better loss
            existing_operators = [v["operator"] for v in data[bm]]
            if operator in existing_operators:
                old_best_value = float(
                    [v["loss/test"] for v in data[bm] if v["operator"] == operator][0]
                )
                if best["loss/test"] >= old_best_value:
                    print(
                        f"Skipping {bm} {operator} because it has a better loss already: "
                        f"old={old_best_value:.4e}  new={best['loss/test']:.4e}"
                    )
                    continue
                else:
                    # Remove the old operator
                    data[bm] = [v for v in data[bm] if v["operator"] != operator]

            # Filter out parameters
            best["params"] = {
                k: v for k, v in best["params"].items() if k not in filter_params
            }

            best["num_params"] = f'{int(best["num_params"])}'
            best["loss/train"] = f'{best["loss/train"]:.2e}'
            best["loss/test"] = f'{best["loss/test"]:.2e}'

            data[bm].append(best)

    with open("results/data.json", "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    process()
