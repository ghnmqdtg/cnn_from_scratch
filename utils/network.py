# ----------------------------------------------------------------------------
# Filename    : network.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ----------------------------------------------------------------------------
import random
import numpy as np
import matplotlib.pyplot as plt
from utils.utils import plot_results, plot_confusion_matrix
from utils.optimizer import SGD
import config


def result_statistics(confusion_matrix):
    """
    Compute the statistics of the results

    Parameters
    ----------
    confusion_matrix : np.ndarray
        The confusion matrix
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


def generate_confusion_matrix(x_train, y_train, network, shuffle=False, sample=-1, plot=False, file_paths=[]):
    """
    Compute the statistics of the results

    Parameters
    ----------
    x_train : list
        List of training data
    y_train : list
        List of training labels
    network : list
        List of network
    shuffle : bool
        To indicate whether to shuffle the training data or not
    sample : int, optional
        Number of samples to draw from the training data. If -1, draw all samples.
    plot : bool
        To plot the confusion matrix or not
    file_paths : list
        List of file paths of training data

    Returns:
    --------
    confusion_matrix : np.array
        Confusion matrix
    """

    files = file_paths.copy()
    inputs = x_train.copy()
    targets = y_train.copy()

    labels = []
    preds = []

    # Shuffle the the lists
    if shuffle:
        fixed_seed = random.random()
        random.Random(fixed_seed).shuffle(files)
        random.Random(fixed_seed).shuffle(inputs)
        random.Random(fixed_seed).shuffle(targets)

    # Predict the samples
    for x, y in zip(inputs[:sample], targets[:sample]):
        output = predict(network, x)
        labels.append(y)
        preds.append(np.argmax(output))

    confusion_matrix = compute_confusion_matrix(labels, preds)

    if plot:
        plot_results(files[:sample], preds, labels)

    return confusion_matrix


def predict(network, input):
    """
    Predicts the output of the network

    Parameters
    ----------
    network : list
        List of network
    input : np.array
        The input to be predicted

    Returns
    -------
    output : int
        The output of the network
    """
    output = input
    for layer in network:
        output = layer.forward(output)
    return output


def train(network, loss, loss_prime, x_train, y_train, epochs=1000, lr=0.01, optimizer="SGD", file_paths=[], verbose=True):
    """
    Compute the statistics of the results

    Parameters
    ----------
    network : list
        List of network
    loss : float
        Loss function
    loss_prime : float
        The backward function of the loss function
    x_train : list
        List of training data
    y_train : list
        List of training labels
    epochs : int
        The number of epochs to train
    lr : float
        The learning rate of the optimizer
    optimizer : str
        The optimizer to use
    file_paths : list
        List of file paths of training data
    verbose : bool
        To print the training information or not
    """
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

    # Plot and save the confusion matrix
    plot_confusion_matrix(confusion_matrix)
    # Plot and save the training history
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Training History', fontsize=20)
    # Loss statistics
    axs[0].set_title("Loss")
    axs[0].plot(history["loss"])
    axs[0].set(xlabel="Epoch", ylabel="MSE")
    # Accuracy statistics
    axs[1].set_title("Accuracy")
    axs[1].plot(history["acc"])
    axs[1].set(xlabel="Epoch", ylabel="Accuracy")

    plt.savefig(f'{config.DST_FOLDER}/history.jpg')

    # Shuffle testing
    confusion_matrix = generate_confusion_matrix(
        x_train, y_train, network, shuffle=True, sample=15, file_paths=file_paths, plot=True)
