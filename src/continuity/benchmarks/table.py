import pandas as pd
from typing import List
import matplotlib.pyplot as plt

default_keys = [
    "epoch",
    "loss/train",
    "loss/test",
    "time/train",
    "time/test",
]


class BenchmarkTable:
    def __init__(
        self,
        keys: List[str] = default_keys,
    ):
        self.keys = keys
        self.all_runs = []

    def add_run_stats(self, stats: dict):
        self.all_runs.append(stats)

    def as_data_frame(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame([r]) for r in self.all_runs])

    def by_benchmark_and_operator(self) -> dict:
        processed = {}
        df = self.as_data_frame()

        benchmarks = df["Benchmark"].unique()

        for bm in benchmarks:
            processed[bm] = {}

            mask = df["Benchmark"] == bm
            df_bm = df.loc[mask]

            keys = self.keys + ["params"]
            groups = df_bm.groupby(["Operator"])[keys]
            processed[bm] = groups.agg("mean")

        return processed

    def generate_loss_plot(self, bm, op) -> str:
        fig, ax = plt.subplots(figsize=(4, 2))
        filename = f"img/{bm}_{op}.png"

        # Plot all loss histories
        loss_history = None
        for r in self.all_runs:
            if r["Benchmark"] == bm and r["Operator"] == op:
                loss_history = r["loss_history"]
                ax.plot(range(len(loss_history)), loss_history, "k-", alpha=0.5)

        ax.axis("off")
        plt.yscale("log")
        fig.savefig(filename)
        return filename

    def write_html(self, filename: str = "results.html"):
        by_benchmark_and_operator = self.by_benchmark_and_operator()

        table = ""
        for bm, benchmark_data in by_benchmark_and_operator.items():
            table += f"<h2>{bm}</h2>\n"

            table += "<table>\n<thead>\n<tr><th></th><th>Params</th><th></th>"
            for key in self.keys:
                table += f"<th>{key}</th>"
            table += "</tr>\n</thead>\n<tbody>\n"

            for op in benchmark_data.index:
                operator_data = benchmark_data[benchmark_data.index == op]

                table += f"<tr><th>{op}</th>"

                params = operator_data["params"].values[0]
                table += f"<td>{params:.3g}</td>"

                loss_plot = self.generate_loss_plot(bm, op)
                table += f'<th><img height="60px" src="{loss_plot}"></th>'

                for key in self.keys:
                    v = operator_data[key].values[0]
                    table += f"<td>{v:.3g}</td>"

                table += "</tr>\n"
            table += "</tbody>\n</table>"

        with open("template.html", "r") as f:
            template = f.read()

        with open(filename, "w") as f:
            html = template.format(table=table)
            f.write(html)
