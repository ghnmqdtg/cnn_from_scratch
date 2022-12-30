# ----------------------------------------------------------------------------
# Filename    : losses.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ----------------------------------------------------------------------------
import numpy as np


def mse(y_true, y_pred):
    """
    Mean squared error

    Parameters
    ----------
    y_true : int or float
        The true value
    y_pred : int or float
        The predicted value

    Returns
    -------
    mse : float
        The mean squared error
    """
    return np.mean(np.power(y_true - y_pred, 2))


def mse_prime(y_true, y_pred):
    """
    The back propagated mean squared error

    Parameters
    ----------
    y_true : int or float
        The true value
    y_pred : int or float
        The predicted value

    Returns
    -------
    mse_prime : float
        The derivative of the mean squared error
    """
    return 2 * (y_pred - y_true) / np.size(y_true)
