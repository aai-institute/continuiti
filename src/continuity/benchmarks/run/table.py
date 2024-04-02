import mlflow
import pandas as pd
import matplotlib.pyplot as plt


class BenchmarkTable:
    def __init__(self):
        self.keys = [
            "loss/train",
            "loss/test",
        ]

        self.client = mlflow.MlflowClient()
        experiments = self.client.search_experiments()
        experiment_ids = [e.experiment_id for e in experiments]
        self.runs = self.client.search_runs(experiment_ids)

    def as_data_frame(self) -> pd.DataFrame:
        all_runs_dicts = []

        for run in self.runs:
            history = self.client.get_metric_history(run.info.run_id, "loss/train")
            loss_history = [m.value for m in history]

            all_runs_dicts.append(
                {
                    "Benchmark": run.data.tags["benchmark"],
                    "Operator": run.data.tags["operator"],
                    "num_params": run.data.metrics["num_params"],
                    "loss/train": run.data.metrics["loss/train"],
                    "loss/test": run.data.metrics["loss/test"],
                    "train_history": loss_history,
                    "max_epochs": run.data.params["max_epochs"],
                    "params": run.data.params,
                }
            )

        return pd.concat([pd.DataFrame([r]) for r in all_runs_dicts])

    def by_benchmark_and_operator(self) -> dict:
        processed = {}
        df = self.as_data_frame()

        benchmarks = df["Benchmark"].unique()

        for bm in benchmarks:
            mask = df["Benchmark"] == bm
            processed[bm] = df.loc[mask]

        return processed

    def generate_loss_plot(self, operator_data) -> str:
        bm = str(operator_data["Benchmark"].values[0])
        op = str(operator_data["Operator"].values[0])

        fig, ax = plt.subplots(figsize=(5, 2))
        filename = f"img/{bm}_{op}.svg"

        max_epochs = int(operator_data["max_epochs"].iloc[0])
        ax.hlines(1e-5, 0, max_epochs, "black", "--")

        # Plot all runs
        train_history = operator_data["train_history"].values
        for i in range(1, len(train_history)):
            ax.plot(range(len(train_history[i])), train_history[i], "k-", alpha=0.1)

        # Plot the best run
        ax.plot(range(len(train_history[0])), train_history[0], "k-")

        ax.axis("off")
        plt.xlim(0, max_epochs)
        plt.yscale("log")
        plt.tight_layout()
        fig.savefig("html/" + filename)
        return filename

    def write_html(self):
        filename = "html/table.html"
        by_benchmark_and_operator = self.by_benchmark_and_operator()

        benchmarks = sorted(list(by_benchmark_and_operator.keys()))

        # Path to the API documentation
        path = "../api/continuity/"

        table = '<link rel="stylesheet" href="style.css">\n'
        for bm in benchmarks:
            benchmark_data = by_benchmark_and_operator[bm]

            table += f'<h2><a href="{path}benchmarks/#continuity.benchmarks.{bm}">{bm}</a></h2>\n'

            table += '<table class="benchmark-table">\n<thead>\n<tr>'
            table += "<th>Operator</th><th>Params</th><th>Learning Curve</th>"

            for key in self.keys:
                table += f"<th>{key}</th>"
            table += "</tr>\n</thead>\n<tbody>\n"

            # Sort by test loss
            benchmark_data = benchmark_data.sort_values("loss/test")

            visited = set()

            for i, op in enumerate(benchmark_data["Operator"]):
                if op in visited:
                    continue
                else:
                    visited.add(op)

                # Show only the best run for each operator
                sorted_data = benchmark_data[benchmark_data["Operator"] == op]
                operator_data = sorted_data.iloc[0]

                params_dict = operator_data["params"]
                exclude_params = ["max_epochs", "seed", "lr", "tol", "batch_size"]
                params_dict = {
                    k: v for k, v in params_dict.items() if k not in exclude_params
                }
                sorted_keys = sorted(params_dict.keys())
                param_str = ", ".join([f"{k}={params_dict[k]}" for k in sorted_keys])
                table += (
                    f'<tr><th><a href="{path}operators/#continuity.operators.{op}" '
                )
                table += f'>{op}</a><div class="div-params">({param_str})</div></th>'

                num_weights = operator_data["num_params"]
                table += f"<td>{int(num_weights)}</td>"

                loss_plot = self.generate_loss_plot(sorted_data)
                table += f'<td width="150px"><img height="60px" src="{loss_plot}"></td>'

                for key in self.keys:
                    v = operator_data[key]
                    if key in ["epoch"]:
                        table += f"<td>{int(v)}</td>"
                    elif key in ["loss/test"] and i == 0:
                        table += f"<td><b>{v:.3g}</b></td>"
                    else:
                        table += f"<td>{v:.3g}</td>"

                table += "</tr>\n"
            table += "</tbody>\n</table>\n"

        with open(filename, "w") as f:
            f.write(table)
