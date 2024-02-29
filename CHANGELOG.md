# CHANGELOG

## 0.1

- Move all content of `__init__.py` files to sub-modules.
- Add `Trainer` class to replace `operator.fit` method.
- Implement `BelNet`.
- Add `Sampler`, `BoxSampler`, `UniformBoxSampler`, and `RegularGridSampler` classes.
- Moved `DataLoader` into the `fit` method of the `Trainer`.
  Therefore, `Trainer.fit` expects an `OperatorDataset` now.
- Add `FunctionOperatorDataset`, `SampledFunctionSet`, `FunctionSet`, `ParameterizedFunction`, `Function` classes.
- Add `function_dataset` example.
- Change `Sine` dataset to use `FunctionOperatorDataset`.

## 0.0.0 (2024-02-22)

- Set up project structure.
- Implement basic functionality.
- Build documentation.
- Create first notebooks.
- Introduce neural operators.
- Add CI/CD.
