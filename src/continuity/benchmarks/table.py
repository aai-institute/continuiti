import pandas as pd
import matplotlib.pyplot as plt
from continuity.benchmarks.database import BenchmarkDatabase


class BenchmarkTable:
    def __init__(self, db: BenchmarkDatabase):
        self.db = db
        self.keys = [
            "loss/train",
            "loss/test",
        ]

    def all_runs(self):
        return self.db.all_runs

    def as_data_frame(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame([r]) for r in self.all_runs()])

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
        filename = "html/index.html"
        by_benchmark_and_operator = self.by_benchmark_and_operator()

        benchmarks = sorted(list(by_benchmark_and_operator.keys()))

        table = ""
        for bm in benchmarks:
            benchmark_data = by_benchmark_and_operator[bm]
            table += f"<h2>{bm}</h2>\n"

            table += (
                "<table>\n<thead>\n<tr><th></th><th>Params</th><th>Learning Curve</th>"
            )
            for key in self.keys:
                table += f"<th>{key}</th>"
            table += "</tr>\n</thead>\n<tbody>\n"

            # Sort by test loss
            benchmark_data = benchmark_data.sort_values("loss/test")

            for i, op in enumerate(benchmark_data["Operator"]):
                operator_data = benchmark_data[benchmark_data["Operator"] == op]

                table += f"<tr><th>{op}</th>"

                params = operator_data["params"]
                table += f"<td>{int(params)}</td>"

                loss_plot = self.generate_loss_plot(operator_data)
                table += f'<th><img height="60px" src="{loss_plot}"></th>'

                for key in self.keys:
                    v = operator_data[key].values[0]
                    if key in ["epoch"]:
                        table += f"<td>{int(v)}</td>"
                    elif key in ["loss/test"] and i == 0:
                        table += f"<td><b>{v:.3g}</b></td>"
                    else:
                        table += f"<td>{v:.3g}</td>"

                table += "</tr>\n"
            table += "</tbody>\n</table>"

        with open("html/template.html", "r") as f:
            template = f.read()

        with open(filename, "w") as f:
            html = template.format(table=table)
            f.write(html)
