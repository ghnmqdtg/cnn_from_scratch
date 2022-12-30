# ----------------------------------------------------------------------------
# Filename    : optimizer.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ----------------------------------------------------------------------------
class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, grad):
        return grad - self.lr * grad
