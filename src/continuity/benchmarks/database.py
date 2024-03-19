import time
import pickle

# TODO: Use sqlite


class BenchmarkDatabase:
    def __init__(self, file: str = "benchmarks.db"):
        self.file = file
        self.all_runs = []
        self.load()

    def add_run(self, stats: dict):
        stats["timestamp"] = time.time()
        self.all_runs.append(stats)
        self.save()

    def load(self):
        try:
            with open(self.file, "rb") as f:
                self.all_runs = pickle.load(f)
            print(f"Load database with {len(self.all_runs)} entries.")
        except FileNotFoundError:
            print(f"Creating new database: {self.file}")

    def save(self):
        with open(self.file, "wb") as f:
            pickle.dump(self.all_runs, f)
