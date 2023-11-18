import torch
import matplotlib.pyplot as plt
from continuity.data.sine import SineWaves
from continuity.plotting.plotting import *
from continuity.model.model import NeuralOperator
from torch.utils.tensorboard import SummaryWriter

# Set random seed
torch.manual_seed(0)


def test_sine():
    # Size of data set
    size = 4

    # Create data set
    dataset = SineWaves(
        num_sensors=32,
        size=size,
        batch_size=32,
    )

    model = NeuralOperator(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        depth=1,
        kernel_width=32,
        kernel_depth=8,
    )

    # Load model
    load = False
    if load:
        log_dir = "runs/<log_dir>"
        model.load_state_dict(torch.load(f"{log_dir}/model_weights.pth"))

    # Train model
    epochs = 100
    writer = SummaryWriter()

    train = True
    if train:
        # Setup optimizer and fit model
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        model.compile(optimizer, criterion)
        model.fit(dataset, epochs, writer)

        torch.save(model.state_dict(), f"{writer.log_dir}/model_weights.pth")

    # Plot
    plot = True
    if plot:
        for i in range(size):
            obs = dataset.get_observation(i)
            plt.cla()
            plot_evaluation(model, dataset, obs)
            plot_observation(obs)
            plot_to_tensorboard(writer, "Plot/train", i)

        # Test observation
        obs = dataset._generate_observation(size)
        plt.cla()
        plot_evaluation(model, dataset, obs)
        plot_observation(obs)
        plot_to_tensorboard(writer, "Plot/validation")


if __name__ == "__main__":
    test_sine()
