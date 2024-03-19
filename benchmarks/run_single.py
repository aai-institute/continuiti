from continuity.benchmarks.runner import BenchmarkRunner, RunConfig
from continuity.benchmarks.database import BenchmarkDatabase
from continuity.benchmarks import SineRegular
from continuity.operators import DeepONet

run = RunConfig(
    benchmark_name="Sine",
    benchmark_factory=lambda: SineRegular(),
    operator_name="DeepONet",
    operator_factory=lambda s: DeepONet(s),
    seed=0,
    lr=1e-4,
    tol=1e-3,
    max_epochs=1000,
    device="cpu",
)

if __name__ == "__main__":
    db = BenchmarkDatabase()
    runner = BenchmarkRunner()

    stats = runner.run(run)
    db.add_run(stats)
