# ----------------------------------------------------------------------------
# Filename    : layers.py
# Created By  : Ting-Wei, Zhang (ghnmqdtg)
# Created Date: 2022/12/23
# version ='1.0'
# ----------------------------------------------------------------------------
import numpy as np
from scipy import signal


class Layer:
    """
    Layer class
    """

    def __init__(self):
        """
        Initialize parameters of weights and biases
        """
        self.input = None
        self.output = None

    def forward(self, input):
        """
        Return the result of layer
        """
        pass

    def backward(self, output_gradient, learning_rate):
        """
        Update the weights and bias of the layer and return the gradient
        """
        pass


class Dense(Layer):
    """
    Dense layer by inheriting from Layer
    """

    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        # W = W - lr * dL/dW
        self.weights -= learning_rate * weights_gradient
        # B = B - lr * dL/dW
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Convolution(Layer):
    """
    Convolution layer by inheriting from Layer
    """

    def __init__(self, input_shape, kernel_size, depth, padding=0):
        input_depth, input_height, input_width = input_shape
        self.input_height = input_height
        self.input_width = input_width
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        # The output shape is (input size + 2 * padding size - kernel size) / (stride) + 1
        # The stride is not defined in this module, so it should be 1 here
        self.output_shape = (depth, input_height + 2 * padding -
                             kernel_size + 1, input_width + 2 * padding - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        # Randomly initialize the parameters of kernels
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.padding = padding

        # For the first CNN layer, use custom kernels
        if self.input_depth * self.depth == 2:
            self.kernel_01 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
            # Randomly initialize the parameters with the range of -1 ~ 1
            self.kernel_02 = np.random.randn(3, 3) - 1
            # Combine the kernels
            self.kernels = np.stack(
                (np.expand_dims(self.kernel_01, axis=0), np.expand_dims(self.kernel_02, axis=0)))
        # For the second CNN layer, use custom kernels
        if self.input_depth * self.depth == 4:
            self.kernel_01 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
            # Randomly initialize the parameters with the range of -1 ~ 1
            self.kernel_02 = np.random.randn(3, 3) - 1
            # Combine the kernels
            self.kernels = np.stack((self.kernel_01, self.kernel_02))
            self.kernels = np.stack((self.kernels, self.kernels))

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        # Get the shape of the padded input
        self.input_padded = np.zeros(
            (self.input_depth, self.input_height + 2 * self.padding, self.input_width + 2 * self.padding))

        # Padding the input
        for d in range(self.input_depth):
            # Pad zeros to the boundaries
            self.input_padded[d] = np.pad(
                self.input[d], self.padding, mode='constant')

        # Update the output
        for i in range(self.depth):
            for j in range(self.input_depth):
                # The output of CNN layer is the cross correlation of padded input and kernels
                self.output[i] += signal.correlate2d(
                    self.input_padded[j], self.kernels[i, j], "valid")

        return self.output

    def backward(self, output_gradient, learning_rate):
        # Initialize the gradient of kernels and input (padded)
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient_padded = np.zeros(
            (self.input_depth, self.input_height + 2 * self.padding, self.input_width + 2 * self.padding))

        for i in range(self.depth):
            for j in range(self.input_depth):
                # kernels gradient is the cross correlation of input and output gradient
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid")
                # Padded input gradient is the cross correlation of output gradient and kernels
                input_gradient_padded[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full")

        # W = W - lr * dL/dW
        self.kernels = self.kernels - learning_rate * kernels_gradient
        # B = B - lr * dL/dW
        self.biases = self.biases - learning_rate * output_gradient
        # Remove the padding
        input_gradient = input_gradient_padded[:,
                                               self.padding:-self.padding, self.padding:-self.padding]
        return input_gradient


class Reshape(Layer):
    """
    Reshapes the input to the given shape.
    """

    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


class MaxPooling:
    """
    Max pooling layer by inheriting from Layer
    """

    def __init__(self, input_shape, pool_shape, stride=2, pad=0, depth=1):
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = self.input_shape
        self.pool_h, self.pool_w = pool_shape
        self.stride = stride
        self.pad = pad
        self.depth = depth
        # Get the shape of the output
        self.out_h = int((self.input_height - self.pool_h) / self.stride + 1)
        self.out_w = int((self.input_width - self.pool_w) / self.stride + 1)

    def patches(self, input):
        """
        Get the patches of the input
        """
        for h in range(self.out_h):
            for w in range(self.out_w):
                patch = input[(h * self.pool_h): (h * self.pool_h + self.pool_h),
                              (w * self.pool_w): (w * self.pool_w + self.pool_w)]
                yield patch, h, w

    def forward(self, input):
        self.input = input
        self.output = np.zeros((self.depth, self.out_h, self.out_w))

        for i in range(self.depth):
            for j in range(self.input_depth):
                # Get the maximum from the patches as the output
                for patch, h, w in self.patches(self.input[j]):
                    self.output[i][h, w] = np.amax(patch, axis=(0, 1))

        return self.output

    def backward(self, output_gradient, learning_rate):
        # Initialize the gradient of input
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                for patch, h, w in self.patches(self.input[j]):
                    # Get the index of maximum of the patches
                    idx_h, idx_w = np.unravel_index(
                        np.argmax(patch, axis=None), patch.shape)
                    # Get the gradient of the maximum of the patches
                    input_gradient[i][h * self.pool_h + idx_h, w *
                                      self.pool_w + idx_w] = output_gradient[i][h, w]
        return input_gradient
