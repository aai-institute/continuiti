from continuity.benchmarks.table import BenchmarkTable
from continuity.benchmarks.database import BenchmarkDatabase


if __name__ == "__main__":
    db = BenchmarkDatabase()

    if len(db) == 0:
        quit()

    table = BenchmarkTable(db)
    table.write_html()
