# CONTRIBUTING TO CONTINUITY

The goal of Continuity is to be a repository of architectures and benchmarks for
operator learning with neural networks, and its applications.

Contributions are welcome from anyone in the form of pull requests,
bug reports and feature requests.

## Local development

This project uses [black](https://github.com/psf/black) to format code and
[pre-commit](https://pre-commit.com/) to invoke it as a git pre-commit hook.

Run the following to set up the pre-commit git hook to run before pushes:

```bash
pre-commit install --hook-type pre-push
```

## Setting up your environment

We strongly suggest using some form of virtual environment for working with the
library. E.g. with venv:

```shell
python -m venv ./venv
. venv/bin/activate
pip install -r docs/requirements.txt
```

## Editable installation

A very convenient way of working with your library during development is to
install it in editable mode into your environment by running

```shell
pip install -e .
```

## Build documentation

API documentation and examples from notebooks are built with
[mkdocs](https://www.mkdocs.org/), with versioning handled by
[mike](https://github.com/jimporter/mike).

Notebooks are an integral part of the documentation as well.

Use the following command to build the documentation the same way it is
done in CI:

```bash
mkdocs build
```

Locally, you can use this command instead to continuously rebuild documentation
on changes to the `docs` and `src` folder:

```bash
mkdocs serve
```

This will rebuild the documentation on changes to `.md` files inside `docs`,
notebooks and python files.

In order to build the documentation locally (which is done as part of the tox
suite) [pandoc](https://pandoc.org/) is required. Except for OSX, it should be
installed automatically as a dependency with `requirements-docs.txt`. Under OSX
you can install pandoc (you'll need at least version 2.11) with:

```shell
brew install pandoc
```

## Testing

Automated builds, tests, generation of documentation and publishing are handled
by [CI pipelines](#CI). Before pushing your changes to the remote we recommend
to execute `pytest` locally in order to detect mistakes early on and to avoid
failing pipelines.

Slow tests (> 5s) are marked by the `@pytest.mark.slow` decorator.
To run all tests except the slow ones, use:

```shell
pytest -m "not slow"
```

## Notebooks

We use notebooks both as documentation (copied over to `docs/examples`) and as
integration tests. All notebooks in the `notebooks` directory are executed
during the test run.

Because we want documentation to include the full dataset, we commit notebooks
with their outputs running with full datasets to the repo. The notebooks are
then added by CI to the section
[Examples](https://aai-institute.github.io/continuity/examples.html) of the
documentation.

### Hiding cells in notebooks

You can isolate boilerplate code into separate cells which are then hidden
in the documentation. In order to do this, cells are marked with tags understood
by the mkdocs plugin
[`mkdocs-jupyter`](https://github.com/danielfrg/mkdocs-jupyter#readme),
namely adding the following to the metadata of the relevant cells:

```yaml
"tags": [
  "hide"
]
```

To hide the cell's input and output.

Or:

```yaml
"tags": [
  "hide-input"
]
```

To only hide the input and

```yaml
"tags": [
  "hide-output"
]
```
for hiding the output only.

### Plots in Notebooks
If you add a plot to a notebook, which should also render nicely in browser
dark mode, add the tag *invertible-output*, i.e.

```yaml
"tags": [
  "invertible-output"
]
```
This applies a simple CSS-filter to the output image of the cell.
