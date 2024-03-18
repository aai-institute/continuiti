import pandas as pd
from continuity.benchmarks.runner import BenchmarkRunner
from continuity.benchmarks.table import BenchmarkTable

from runs import all_runs

if __name__ == "__main__":
    table = BenchmarkTable()
    runner = BenchmarkRunner()

    df = pd.DataFrame()
    for run in all_runs():
        stats = runner.run(run)
        table.add_run_stats(stats)

    table.write_html()
