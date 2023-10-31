import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
from data import SineWaves
from plotting import plot_observation, plot_evaluation
from model import FullyConnectedModel
from dadaptation import DAdaptSGD

# Set random seed
torch.manual_seed(0)

def main():
    # Size of data set
    size = 1

    # Create data set
    dataset = SineWaves(32, size, batch_size=1)

    # Create model
    model = FullyConnectedModel(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        num_sensors=dataset.num_sensors,
        width=128,
        depth=32,
    )

    # Create log dir
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = f"./logs/{timestamp}"
    print(f"Logging to {log_dir}")
    os.makedirs(log_dir)
       
    # Load model
    load = False
    if load:
        model.load_state_dict(torch.load('model_weights.pth'))
    
    # Train model
    epochs = 1000

    train = True
    if train:
        # Setup optimizer and fit model
        optimizer = DAdaptSGD(model.parameters())
        criterion = torch.nn.MSELoss()

        model.compile(optimizer, criterion)
        model.fit(dataset, epochs)

        torch.save(model.state_dict(), 'model_weights.pth')

    # Plot
    plot = True
    if plot:
        for i in range(size):
            obs = dataset.get_observation(i)
            plt.cla()
            plot_evaluation(model, dataset, obs)
            plot_observation(obs)
            plt.savefig(f"{log_dir}/plot_{i}.png")

        # Test observation
        obs = dataset._generate_observation(0.5)
        plt.cla()
        plot_evaluation(model, dataset, obs)
        plot_observation(obs)
        plt.savefig(f"{log_dir}/plot_test.png")


if __name__ == '__main__':
    main()