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
            processed[bm] = {}

            mask = df["Benchmark"] == bm
            df_bm = df.loc[mask]

            # Find the best run w.r.t. test loss
            groups = df_bm.groupby(["Operator"])
            idx = groups["loss/test"].transform(min) == df_bm["loss/test"]

            processed[bm] = df_bm[idx]

        return processed

    def generate_loss_plot(self, operator_data) -> str:
        bm = str(operator_data["Benchmark"].values[0])
        op = str(operator_data["Operator"].values[0])

        fig, ax = plt.subplots(figsize=(5, 2))
        filename = f"img/{bm}_{op}.png"

        max_epochs = int(operator_data["max_epochs"])
        ax.hlines(1e-5, 0, max_epochs, "black", "--")

        train_history = operator_data["train_history"].values[0]
        ax.plot(range(len(train_history)), train_history, "k-")

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

            for i, op in enumerate(benchmark_data["Operator"]):
                operator_data = benchmark_data[benchmark_data["Operator"] == op]

                params_dict = operator_data["params"][0]
                exclude_params = ["max_epochs", "seed", "lr", "tol", "batch_size"]
                params_dict = {
                    k: v for k, v in params_dict.items() if k not in exclude_params
                }
                param_str = " ".join([f"{k}={v} " for k, v in params_dict.items()])
                table += (
                    f'<tr><th><a href="{path}operators/#continuity.operators.{op}" '
                )
                table += f'title="{param_str}">{op}</a></th>'

                num_weights = operator_data["num_params"]
                table += f"<td>{int(num_weights)}</td>"

                loss_plot = self.generate_loss_plot(operator_data)
                table += f'<td width="150px"><img height="60px" src="{loss_plot}"></td>'

                for key in self.keys:
                    v = operator_data[key].values[0]
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
