class SGD:
    """
    Stochastic Gradient Descent
    """

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, grad):
        return grad - self.lr * grad
