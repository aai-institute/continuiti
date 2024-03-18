import pandas as pd


class BenchmarkTable:
    def __init__(self):
        self.all_runs = []

    def add_run_stats(self, stats: dict):
        self.all_runs.append(stats)

    def as_data_frame(self) -> pd.DataFrame:
        return pd.concat([pd.DataFrame([r]) for r in self.all_runs])

    def by_benchmark_and_operator(self) -> dict:
        processed = {}
        df = self.as_data_frame()

        benchmarks = df["Benchmark"].unique()
        operators = df["Operator"].unique()

        # Keys to compute mean values
        keys = [
            "loss/train",
            "loss/test",
            "epoch",
            "num_params",
            "time/train",
            "time/test",
        ]

        for bm in benchmarks:
            processed[bm] = {}

            mask = df["Benchmark"] == bm
            df_bm = df.loc[mask]

            for op in operators:
                processed[bm][op] = {}

                mask = df_bm["Operator"] == op
                df_bm_op = df_bm[mask]

                mean = {}
                for key in keys:
                    mean[key] = df_bm_op[key].mean()

                processed[bm][op] = mean

        return processed

    def write_html(self, filename: str = "results.html"):
        data: dict = self.by_benchmark_and_operator()

        table = ""
        for bm in data:
            table += f"<h2>{bm}</h2>\n  "
            df = pd.DataFrame(data[bm]).T
            table += df.to_html()

        with open("template.html", "r") as f:
            template = f.read()

        with open(filename, "w") as f:
            html = template.format(table=table)
            f.write(html)
