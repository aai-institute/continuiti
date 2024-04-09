import matplotlib.pyplot as plt
from continuity.benchmarks import Benchmark, NavierStokes
from continuity.data import OperatorDataset


def test_navierstokes_return_type_correct():
    benchmark = NavierStokes()
    assert isinstance(benchmark.train_dataset, OperatorDataset)
    assert isinstance(benchmark.test_dataset, OperatorDataset)


def test_navierstokes_can_initialize_default():
    assert isinstance(NavierStokes(), Benchmark)


def test_navierstokes_shapes_and_plot():
    benchmark = NavierStokes()
    assert len(benchmark.train_dataset) == 1000
    assert len(benchmark.test_dataset) == 200
    for dataset in [benchmark.train_dataset, benchmark.test_dataset]:
        for x, u, y, v in dataset:
            assert x.shape == (64 * 64 * 10, 3)
            assert u.shape == (64 * 64 * 10, 1)
            assert y.shape == (64 * 64 * 10, 3)
            assert v.shape == (64 * 64 * 10, 1)

    fig, axs = plt.subplots(1, 2, subplot_kw={'projection': '3d'}, figsize=(10, 5))
    x, u, y, v = benchmark.test_dataset[0]
    axs[0].scatter(x[:, 2], x[:, 0], x[:, 1], s=1, c=u, cmap='jet', alpha=0.7)
    axs[1].scatter(y[:, 2], y[:, 0], y[:, 1], s=1, c=v, cmap='jet', alpha=0.7)
    for i in range(2):
        axs[i].set_xlabel('t')
        axs[i].set_ylabel('x')
        axs[i].set_zlabel('y')
    axs[0].set_title('Input')
    axs[1].set_title('Output')

    try:
        fig.savefig('docs/benchmarks/navierstokes.png', dpi=500)
    except FileNotFoundError:
        pass


