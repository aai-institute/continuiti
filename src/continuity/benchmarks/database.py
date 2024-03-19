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

    def __len__(self):
        return len(self.all_runs)

    def load(self):
        try:
            with open(self.file, "rb") as f:
                self.all_runs = pickle.load(f)
            print(f"Load database with {len(self.all_runs)} entries.")
        except FileNotFoundError:
            pass

    def save(self):
        with open(self.file, "wb") as f:
            pickle.dump(self.all_runs, f)
