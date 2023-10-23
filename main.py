import matplotlib.pyplot as plt
from data import Flame
from plotting import plot_observation, plot_evaluation
from model import ContinuityModel
import keras_core as keras
from datetime import datetime

# Set random seed
keras.utils.set_random_seed(316)

def main():
    # Size of data set
    size = 1

    # Create data set
    dataset = Flame(size, batch_size=32)
    coordinate_dim = dataset.coordinate_dim
    num_channels = dataset.num_channels

    # Create model
    model = ContinuityModel(
        coordinate_dim=coordinate_dim,
        num_channels=num_channels,
        num_sensors=dataset.num_sensors,
        width=2048,
        depth=128,
    )
    
    # Train model
    load = False
    if load:
        model.load_weights("model.weights.h5")
    else:
        optimizer = keras.optimizers.SGD(learning_rate=1e-8)
        model.compile(loss="mse", optimizer=optimizer)
    
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        model.fit(dataset, epochs=100, callbacks=[
            keras.callbacks.TensorBoard(log_dir=f'./logs/{timestamp}'),
            keras.callbacks.LearningRateScheduler(lambda _, lr: lr * 0.999)
        ])
        model.save_weights("model.weights.h5")

    # Plot
    for i in range(size):
        obs = dataset.get_observation(i)
        plt.cla()
        plot_evaluation(model, dataset, obs)
        plot_observation(obs)
        plt.savefig(f"plot_{i}.png")

if __name__ == '__main__':
    main()