# CHANGELOG

## 0.1

- Move all content of `__init__.py` files to sub-modules.
- Add `Trainer` class to replace `operator.fit` method.
- Implement `BelNet`.
- Add `Sampler`, `BoxSampler`, `UniformBoxSampler`, and `RegularGridSampler` classes.
- Moved `DataLoader` into the `fit` method of the `Trainer`.
  Therefore, `Trainer.fit` expects an `OperatorDataset` now.
- A `Criterion` now enables stopping the training loop.
- The `plotting` module has been removed.
- Add `timeseries.ipynb` example.
- Add `Function`, `FunctionSet`, and `FunctionOperatorDataset` classes.
- Add `function.ipynb` example.
- Add `Benchmark` base class.
- Add `SineBenchmark`.
- Implement `DeepNeuralOperator`.
- Generalize `NeuralOperator` to take a list of operators.
- The `data.DatasetShapes` class becomes `operators.OperatorShapes` without `num_observations` attribute.
- Change `torch` dependency from "==2.1.0" to ">=2.1.0,<3.0.0".
- Change `optuna` dependency from "3.5.0" to ">=3.5.0,<4.0.0".
- Add `FourierLayer` and `FourierNeuralOperator` with example.
- Add `benchmarks` infrastructure.
- An `Operator` now takes a `device` argument.
- Add `QuantileScaler` class.

## 0.0.0 (2024-02-22)

- Set up project structure.
- Implement basic functionality.
- Build documentation.
- Create first notebooks.
- Introduce neural operators.
- Add CI/CD.
