import numpy as np
from scipy import signal


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass


class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias

    def backward(self, output_gradient, learning_rate):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient


class Convolution(Layer):
    def __init__(self, input_shape, kernel_size, depth, padding=0):
        input_depth, input_height, input_width = input_shape
        self.input_height = input_height
        self.input_width = input_width
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height + 2 * padding -
                             kernel_size + 1, input_width + 2 * padding - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)
        self.biases = np.random.randn(*self.output_shape)
        self.padding = padding

        if self.input_depth * self.depth == 2:
            self.kernel_01 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
            self.kernel_02 = np.random.randn(3, 3) - 1
            self.kernels = np.stack(
                (np.expand_dims(self.kernel_01, axis=0), np.expand_dims(self.kernel_02, axis=0)))
        if self.input_depth * self.depth == 4:
            self.kernel_01 = np.array([[-1, -1, 1], [-1, 0, 1], [-1, 1, 1]])
            self.kernel_02 = np.random.randn(3, 3) - 1
            self.kernels = np.stack((self.kernel_01, self.kernel_02))
            self.kernels = np.stack((self.kernels, self.kernels))

    def forward(self, input):
        self.input = input
        self.input_padded = np.zeros(
            (self.input_depth, self.input_height + 2 * self.padding, self.input_width + 2 * self.padding))

        for d in range(self.input_depth):
            # Pad zeros to the boundaries
            self.input_padded[d] = np.pad(
                self.input[d], self.padding, mode='constant')

        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(
                    self.input_padded[j], self.kernels[i, j], "valid")
        return self.output

    def backward(self, output_gradient, learning_rate):
        kernels_gradient = np.zeros(self.kernels_shape)
        input_gradient_padded = np.zeros(
            (self.input_depth, self.input_height + 2 * self.padding, self.input_width + 2 * self.padding))

        for i in range(self.depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(
                    self.input[j], output_gradient[i], "valid")
                input_gradient_padded[j] += signal.convolve2d(
                    output_gradient[i], self.kernels[i, j], "full")

        self.kernels = self.kernels - learning_rate * kernels_gradient
        self.biases = self.biases - learning_rate * output_gradient

        input_gradient = input_gradient_padded[:,
                                               self.padding:-self.padding, self.padding:-self.padding]
        return input_gradient


class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient, learning_rate):
        return np.reshape(output_gradient, self.input_shape)


class MaxPooling:
    def __init__(self, input_shape, pool_shape, stride=2, pad=0, depth=1):
        self.input_shape = input_shape
        self.input_depth, self.input_height, self.input_width = self.input_shape
        self.pool_h, self.pool_w = pool_shape
        self.stride = stride
        self.pad = pad
        self.depth = depth

        self.x = None
        self.arg_max = None

    def patches(self, input):
        for h in range(self.out_h):
            for w in range(self.out_w):
                patch = input[(h * self.pool_h): (h * self.pool_h + self.pool_h),
                              (w * self.pool_w): (w * self.pool_w + self.pool_w)]
                yield patch, h, w

    def forward(self, input):
        self.input = input
        self.out_h = int(1 + (self.input_height - self.pool_h) / self.stride)
        self.out_w = int(1 + (self.input_width - self.pool_w) / self.stride)
        output = np.zeros((self.depth, self.out_h, self.out_w))

        for i in range(self.depth):
            for j in range(self.input_depth):
                for patch, h, w in self.patches(self.input[j]):
                    output[i][h, w] = np.amax(patch, axis=(0, 1))

        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                for patch, h, w in self.patches(self.input[j]):
                    patch_h, patch_w = patch.shape
                    max_val = np.amax(patch, axis=(0, 1))

                    for idx_h in range(patch_h):
                        for idx_w in range(patch_w):
                            if patch[idx_h, idx_w] == max_val:
                                input_gradient[i][h*self.pool_h+idx_h, w *
                                                  self.pool_w+idx_w] = output_gradient[i][h, w]
        return input_gradient
