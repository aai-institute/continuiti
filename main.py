import matplotlib.pyplot as plt
from data import SineWaves
from plotting import plot_observation, plot_evaluation
from model import FNNModel
import keras_core as keras

def main():
    # Number of sensors in observation
    num_sensors = 11

    # Size of data set
    size = 1

    # Create data set
    dataset = SineWaves(num_sensors, size)

    # Create model
    model = FNNModel(
        coordinate_dim=dataset.coordinate_dim,
        num_channels=dataset.num_channels,
        num_sensors=num_sensors,
        width=128,
        depth=32,
    )

    # Train model
    load = False
    if load:
        model.predict(dataset)
        model.load_weights("model.weights.h5")
    else:
        adam = keras.optimizers.Adam(learning_rate=1e-3)
        model.compile(loss="mse", optimizer=adam)
        model.fit(dataset, epochs=1000)
        model.save_weights("model.weights.h5")

    # Plot
    obs = dataset.get_observation(0)
    plot_observation(obs)

    obs.sensors[3] = obs.sensors[4] # Mask some sensor
    plot_evaluation(model, dataset, obs)
    plt.savefig('plot.png')

if __name__ == '__main__':
    main()