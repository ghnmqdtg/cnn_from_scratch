import numpy as np
import random as rd
import matplotlib.pyplot as plt
from utils.utils import plot_results, plot_confusion_matrix
from utils.optimizer import SGD
import config


def result_statistics(confusion_matrix):
    """
    Compute the statistics of the results
    """
    acc = (confusion_matrix[0, 0] +
           confusion_matrix[1, 1]) / confusion_matrix.sum()
    return acc


def compute_confusion_matrix(true, pred):
    '''
    Computes a confusion matrix using numpy for two np.arrays true and pred.

    Parameters
    ----------
    true : list
        List of true answers
    pred : list
        List of predictions
    '''

    K = len(np.unique(true))  # Number of classes
    result = np.zeros((K, K))

    for i in range(len(true)):
        result[true[i]][pred[i]] += 1

    return result


def generate_confusion_matrix(X, Y, network, shuffle=False, sample=-1, plot=False, file_paths=[]):
    """
    Compute the statistics of the results
    """
    files = file_paths.copy()
    inputs = X.copy()
    targets = Y.copy()

    labels = []
    preds = []

    if shuffle:
        fixed_seed = rd.random()
        rd.Random(fixed_seed).shuffle(files)
        rd.Random(fixed_seed).shuffle(inputs)
        rd.Random(fixed_seed).shuffle(targets)

    for file_path, x, y in zip(files[:sample], inputs[:sample], targets[:sample]):
        output = predict(network, x)
        labels.append(y)
        preds.append(np.argmax(output))
        # print(f"preds: {np.argmax(output)}, labels: {y}")

    confusion_matrix = compute_confusion_matrix(labels, preds)

    if plot:
        plot_results(files, preds, labels)

    return confusion_matrix


def predict(network, input):
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, lr=0.01, optimizer="SGD", verbose=True, file_paths=[]):
    confusion_matrix = []
    history = {
        "acc": [],
        "loss": []
    }

    if optimizer == "SGD":
        opt = SGD(lr)

    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            grad = opt.update(grad)

            for layer in reversed(network):
                grad = layer.backward(grad, lr)

        error /= len(x_train)

        confusion_matrix = generate_confusion_matrix(
            x_train, y_train, network, shuffle=False, sample=-1, file_paths=file_paths)
        acc = result_statistics(confusion_matrix)

        history["loss"].append(error)
        history["acc"].append(acc)

        if verbose:
            print(f"{e + 1}/{epochs}, loss={error}, acc={acc}")

    plot_confusion_matrix(confusion_matrix)
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Training Statistics', fontsize=20)

    # Loss statistics
    axs[0].set_title("Loss")
    axs[0].plot(history["loss"])
    axs[0].set(xlabel="Epoch")
    # Accuracy statistics
    axs[1].set_title("Accuracy")
    axs[1].plot(history["acc"])
    axs[1].set(xlabel="Epoch")

    plt.savefig(f'{config.DST_FOLDER}/history.jpg')

    # Shuffle testing
    confusion_matrix = generate_confusion_matrix(
        x_train, y_train, network, shuffle=True, sample=15, file_paths=file_paths, plot=True)
