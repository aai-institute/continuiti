import pytest
from continuity.benchmarks.run import BenchmarkRunner, RunConfig
from continuity.benchmarks import SineRegular
from continuity.operators import DeepNeuralOperator


@pytest.mark.slow
def test_runner():
    config = RunConfig(
        benchmark_factory=SineRegular,
        operator_factory=DeepNeuralOperator,
        max_epochs=100,
    )
    BenchmarkRunner.run(config)
