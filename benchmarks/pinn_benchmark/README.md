# PINN-BENCHMARK

Benchmarking the performance of physics-informed neural networks (PINNs).

## Installation

```
conda activate pinn-env

conda install pytorch
conda install -c conda-forge deepxde
conda install -c conda-forge fenics-dolfinx
```

For Apple Silicon, use `conda install nomkl` before installing `pytorch`.


## Results

#### Poisson equation on a unit square
![poisson](https://github.com/aai-institute/Continuity/blob/main/benchmarks/pinn_benchmark/output/error_vs_wall_time_poisson.png?raw=true)
