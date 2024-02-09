<div align="center">

<img alt="Continuity" src="https://aai-institute.github.io/continuity/img/icon.png" width="100">

<h1>Continuity</h1>

Learning function operators with neural networks.

<a href="https://pytorch.org/get-started/locally/">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white">
</a>
<a href="https://aai-institute.github.io/continuity/">
  <img alt="Documentation" src="https://img.shields.io/badge/Documentation-blue">
</a>
<a href="https://github.com/aai-institute/continuity/actions/workflows/test.yml">
  <img alt="Test" src="https://github.com/aai-institute/continuity/actions/workflows/test.yml/badge.svg">
</a>
</div>

**Continuity** is a Python package for machine learning on function operators.
It implements various neural operator architectures (e.g., DeepONets),
physics-informed loss functions to train based on PDEs, and a collection of
examples and benchmarks.

## Installation
Clone the repository and install the package using pip.
```
git clone https://github.com/aai-institute/continuity.git
cd continuity
pip install -e .
```

## Usage
Our [Documentation](https://aai-institute.github.io/continuity/) contains a
verbose introduction to operator learning, a collection of examples using
Continuity, and a class documentation.

In general, the operator syntax in Continuity is
```python
v = operator(x, u(x), y)
```
mapping a function `u` (evaluated at `x`) to function `v` (evaluated in `y`).
For more details, see
[Learning Operators](https://aai-institute.github.io/continuity/operators/index.html).

## Contributing
Contributions are welcome from anyone in the form of pull requests, bug reports
and feature requests. If you find a bug or have a feature request, please open
an issue on GitHub. If you want to contribute code, please fork the repository
and submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for details on
local development.

## License
This project is licensed under the GNU LGPLv3 License - see the
[LICENSE](LICENSE) file for details.
