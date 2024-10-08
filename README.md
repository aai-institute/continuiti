<div align="center">
<img alt="continuiti" src="https://aai-institute.github.io/continuiti/img/icon.png" width="100">

<h1>continuiti</h1>

Learning function operators with neural networks.

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Documentation](https://img.shields.io/badge/Documentation-blue)](https://aai-institute.github.io/continuiti/)
[![Test](https://github.com/aai-institute/continuiti/actions/workflows/test.yml/badge.svg)](https://github.com/aai-institute/continuiti/actions/workflows/test.yml)
</div>

**continuiti** is a Python package for deep learning on function operators with
a focus on elegance and generality. It provides a _unified interface_ for neural
operators (such as DeepONet or FNO) to be used in a plug and play fashion. As
operator learning is particularly useful in scientific machine learning,
**continuiti** also includes physics-informed loss functions and a collection of
relevant benchmarks.


## Installation
Install the package using pip:
```shell
pip install continuiti
```

Or install the latest development version from the repository:

```
git clone https://github.com/aai-institute/continuiti.git
cd continuiti
pip install -e ".[dev]"
```

## Usage

Our [Documentation](https://aai-institute.github.io/continuiti/) contains a
collection of tutorials on how to learn operators using continuiti,
a collection of how-to guides to solve specific problems, more background,
and a class documentation.

In general, the operator syntax in **continuiti** is
```python
v = operator(x, u(x), y)
```
mapping a function `u` (evaluated at `x`) to function `v` (evaluated in `y`).


## Examples

<div style="text-align: center;">
<a href="https://aai-institute.github.io/continuiti/latest/benchmarks/#navierstokes">
Fourier Neural Operator (FNO) for Navier-Stokes flow<br>
<img alt="navierstokes" src="https://aai-institute.github.io/continuiti/latest/img/ns.png" width="60%">
</a>
</div>

<div style="text-align: center;">
<a href="https://aai-institute.github.io/continuiti/latest/how-to-guides/physicsinformed/">
Physics-informed DeepONet<br>
<img alt="pi-deeponet" src="docs/img/pi-deeponet.png" width="60%">
</a>
</div>

## Contributing

Contributions are welcome from anyone in the form of pull requests, bug reports
and feature requests. If you find a bug or have a feature request, please open
an issue on GitHub. If you want to contribute code, please fork the repository
and submit a pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for details on
local development.


## License

This project is licensed under the GNU LGPLv3 License - see the
[LICENSE](LICENSE) file for details.
