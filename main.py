import torch
import matplotlib.pyplot as plt
from data import SineWaves, Flame
from plotting import *
from model import FullyConnected, DeepONet, NeuralOperator
from dadaptation import DAdaptSGD
from torch.utils.tensorboard import SummaryWriter

# Set random seed
torch.manual_seed(0)

def main():
    # Size of data set
    size = 2

    # Create data set
    # dataset = SineWaves(32, size, batch_size=1)
    dataset = Flame(size, batch_size=1)

    # Create model
    # model = FullyConnected(
    #     coordinate_dim=dataset.coordinate_dim,
    #     num_channels=dataset.num_channels,
    #     num_sensors=dataset.num_sensors,
    #     width=128,
    #     depth=128,
    # )
    # model = DeepONet(
    #     coordinate_dim=dataset.coordinate_dim,
    #     num_channels=dataset.num_channels,
    #     num_sensors=dataset.num_sensors,
    #     branch_width=32,
    #     branch_depth=4,
    #     trunk_width=128,
    #     trunk_depth=32,
    #     basis_functions=4,
    # )
    model = NeuralOperator(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        num_sensors=dataset.num_sensors,
        layers=16,
        kernel_width=32,
        kernel_depth=4,
    )

    # Load model
    load = False
    if load:
        log_dir = 'runs/<log_dir>'
        model.load_state_dict(torch.load(f'{log_dir}/model_weights.pth'))
    
    # Train model
    epochs = 1000
    writer = SummaryWriter()

    train = True
    if train:
        # Setup optimizer and fit model
        # optimizer = DAdaptSGD(model.parameters())
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()

        model.compile(optimizer, criterion)
        model.fit(dataset, epochs, writer)

        torch.save(model.state_dict(), f'{writer.log_dir}/model_weights.pth')

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


if __name__ == '__main__':
    main()