import matplotlib.pyplot as plt
from data import SineWaves
from plotting import plot_observation, plot_evaluation
from model import ContinuityModel
import keras_core as keras
from datetime import datetime
from dadaptation import DAdaptSGD

# Set random seed
keras.utils.set_random_seed(316)

def main():
    # Size of data set
    size = 32

    # Create data set
    dataset = SineWaves(32, size, batch_size=1)
    coordinate_dim = dataset.coordinate_dim
    num_channels = dataset.num_channels

    # Create model
    model = ContinuityModel(
        coordinate_dim=coordinate_dim,
        num_channels=num_channels,
        num_sensors=dataset.num_sensors,
        width=64,
        depth=32,
    )

    # Load model
    load = False
    if load:
        # Build model and load weights
        model(dataset[0][0])
        model.load_weights("model.weights.h5")
    
    # Train model
    train = True
    if train:
        # Setup optimizer and fit model
        optimizer = DAdaptSGD(model.parameters())
        model.compile(optimizer)
    
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_dir = f"./logs/{timestamp}"
        model.fit(dataset, epochs=100, shuffle=True, callbacks=[
            keras.callbacks.TensorBoard(log_dir=log_dir),
        ])
        model.save_weights("model.weights.h5")

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