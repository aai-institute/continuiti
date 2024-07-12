import pytest
from continuiti.benchmarks.sine import SineBenchmark
from continuiti.trainer import Trainer
from continuiti.operators.losses import MSELoss


@pytest.mark.slow
def test_modulus_fno():
    try:
        from continuiti.operators.modulus import FNO
    except ImportError:
        pytest.skip("NVIDIA Modulus not found!")

    # Data set
    benchmark = SineBenchmark(n_train=1)
    dataset = benchmark.train_dataset

    # Operator
    #   Configured like the default continuiti `FourierNeuralOperator`
    #   with depth=3 and width=3 as in `test_fno.py`.
    operator = FNO(
        dataset.shapes,
        decoder_layers=1,
        decoder_layer_size=1,
        decoder_activation_fn="identity",
        num_fno_layers=3,  # "depth" in FourierNeuralOperator
        latent_channels=3,  # "width" in FourierNeuralOperator
        num_fno_modes=dataset.shapes.u.size[0] // 2 + 1,
        padding=0,
        coord_features=False,
    )

    # Train
    Trainer(operator, device="cpu").fit(dataset, tol=1e-12, epochs=10_000)

    # Check solution
    x, u, y, v = dataset.x, dataset.u, dataset.y, dataset.v
    assert MSELoss()(operator, x, u, y, v) < 1e-12


# SineBenchmark(n_train=1024, n_sensors=128, n_evaluations=128), epochs=100

# NVIDIA Modulus FNO
# Parameters: 3560  Device: cpu
# Epoch 100/100  Step 32/32  [====================]  6ms/step  [0:19min<0:00min] - loss/train = 6.3876e-05

# continuiti FNO
# Parameters: 3556  Device: cpu
# Epoch 100/100  Step 32/32  [====================]  3ms/step  [0:10min<0:00min] - loss/train = 1.4440e-04

# -> continuiti FNO is 2x faster than NVIDIA Modulus FNO
# -> NVIDIA Modulus FNO can not handle different number of sensors and evaluations
