import pytest
from continuiti.benchmarks.run import BenchmarkRunner, RunConfig
from continuiti.benchmarks import SineRegular
from continuiti.operators import DeepNeuralOperator


@pytest.mark.slow
def test_runner():
    config = RunConfig(
        benchmark_factory=SineRegular,
        operator_factory=DeepNeuralOperator,
        max_epochs=2,
        batch_size=128,
    )
    BenchmarkRunner.run(config)
