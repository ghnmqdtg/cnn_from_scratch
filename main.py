import numpy as np
import utils.utils as utils
from utils.network import train
from utils.layers import Dense, Convolution, Reshape, MaxPooling
from utils.activations import Sigmoid, Softmax
from utils.losses import mse, mse_prime
from utils.dataset_generator import DatasetGenerator
import config


def model():
    return [
        # Input (1, 8, 8), output (2, 8, 8)
        Convolution((1, 8, 8), 3, 2, 1),
        # Input (2, 8, 8), output (2, 4, 4)
        MaxPooling((2, 8, 8), (2, 2), 2, 0, 2),
        Sigmoid(),
        # Input (2, 4, 4), output (2, 4, 4)
        Convolution((2, 4, 4), 3, 2, 1),
        # Input (4, 2, 2), output (2, 2, 2)
        MaxPooling((2, 4, 4), (2, 2), 2, 0, 2),
        Sigmoid(),
        # Input (2, 2, 2), output (8, 1)
        Reshape((2, 2, 2), (2 * 2 * 2, 1)),
        # Input (8, 1), output (8, 1)
        Dense(2 * 2 * 2, 2),
    ]


if __name__ == '__main__':
    # 81
    np.random.seed(2)

    utils.create_folder(config.DST_FOLDER)

    dataset_generator = DatasetGenerator(config.SRC_FOLDER)
    X, Y = dataset_generator.prepare()
    network = model()

    train(
        network,
        mse,
        mse_prime,
        X,
        Y,
        epochs=1000,
        lr=0.001,
        optimizer="SGD",
        file_paths=dataset_generator.file_paths
    )
