import numpy as np
from NeuralNetwork.AbstractClasses import Layer

"""
Layers
Common data types used in the layers:
    input_shape: tuple - shape of the input - (channels, height, width)
    output_shape: tuple - shape of the output - (num_filters, out_height, out_width)
"""

# A fully connected layer: each neuron connects to all neurons in the next layer
class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape):
        """
        Initialisation of the fully connected layer.
        Arguments:
            input_shape: tuple - shape of the input
            output_shape: tuple - shape of the output
        """
        super().__init__(input_shape, output_shape)
        scale = np.sqrt(2.0 / (input_shape[0] + output_shape[0]))
        self.weights = np.random.randn(output_shape[0], input_shape[0]) * scale
        self.bias = np.zeros((output_shape[0], 1)) 
        
        # momentum terms are used to optimise the training process
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
        
    def forward(self, input):
        """
        Forward pass of the fully connected layer.
        Arguments:
            input: (batch_size, input_shape)
        Returns:
            output: (batch_size, output_shape)
        """
        self.input = input  # cache input for backward pass
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient, learning_rate, momentum=0.9):
        """
        Backward pass of the fully connected layer.
        Arguments:
            output_gradient: (batch_size, output_shape) - gradient of the loss function with respect to the output
            learning_rate: float - learning rate for the parameter updates
            momentum: float - coefficient for the parameter updates
        Returns:
            input_gradient: (batch_size, input_shape)
        """
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        self.weight_momentum = momentum * self.weight_momentum + (1 - momentum) * weights_gradient
        self.bias_momentum = momentum * self.bias_momentum + (1 - momentum) * output_gradient
        return input_gradient

# A 2D convolutional layer: each neuron connects to a local region (kernel) in the input
class Conv2DLayer(Layer):
    def __init__(self, input_shape, num_filters, kernel_size, stride=1, padding=0):
        """
        Initialisation of the 2D convolutional layer.
        Arguments:
            input_shape: tuple - shape of the input
            num_filters: int - number of filters
            kernel_size: int - size of the kernel
            stride: int - stride of the convolution
            padding: int - padding of the convolution
        """
        channels, height, width = input_shape
        out_height = ((height + 2*padding - kernel_size) // stride) + 1
        out_width = ((width + 2*padding - kernel_size) // stride) + 1
        output_shape = (num_filters, out_height, out_width)
        
        super().__init__(input_shape, output_shape)
        
        # He initialization
        scale = np.sqrt(2.0 / (channels * kernel_size * kernel_size))
        self.weights = np.random.randn(num_filters, channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((num_filters, 1, 1))  # Changed shape for proper broadcasting
        
        # Add momentum terms
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, input):
        """
        Forward pass of the 2D convolutional layer.
        Arguments:
            input: (batch_size, input_shape)
        Returns:
            output: (batch_size, output_shape)
        """
        self.input = input
        batch_size, channels, height, width = input.shape
        
        out_height = ((height + 2*self.padding - self.kernel_size) // self.stride) + 1
        out_width = ((width + 2*self.padding - self.kernel_size) // self.stride) + 1
        output = np.zeros((batch_size, self.output_shape[0], out_height, out_width))
        
        # Add padding if needed
        if self.padding > 0:
            self.padded_input = np.pad(input, 
                                     ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), 
                                     mode='constant')
        else:
            self.padded_input = input
        
        # Convolution operation
        for b in range(batch_size):
            for f in range(self.output_shape[0]):
                for i in range(out_height):
                    for j in range(out_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        patch = self.padded_input[b, :, h_start:h_end, w_start:w_end]
                        output[b, f, i, j] = np.sum(patch * self.weights[f]) + self.bias[f]
        
        return output

    def backward(self, output_gradient, learning_rate, momentum=0.9):
        """
        Backward pass of the 2D convolutional layer.
        Arguments:
            output_gradient: (batch_size, output_shape) - gradient of the loss function with respect to the output
            learning_rate: float - learning rate for the parameter updates
            momentum: float - coefficient for the parameter updates
        Returns:
            input_gradient: (batch_size, input_shape)
        """
        batch_size = output_gradient.shape[0]
        
        # Initialize gradients
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)
        input_gradient_padded = np.zeros_like(self.padded_input)
        
        # Compute gradients
        for b in range(batch_size):
            for f in range(self.output_shape[0]):
                for i in range(self.output_shape[1]):
                    for j in range(self.output_shape[2]):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        
                        patch = self.padded_input[b, :, h_start:h_end, w_start:w_end]
                        weights_gradient[f] += patch * output_gradient[b, f, i, j]
                        bias_gradient[f] += output_gradient[b, f, i, j]
                        input_gradient_padded[b, :, h_start:h_end, w_start:w_end] += \
                            self.weights[f] * output_gradient[b, f, i, j]
        
        # Remove padding from input_gradient if padding was added
        if self.padding > 0:
            input_gradient = input_gradient_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            input_gradient = input_gradient_padded
        
        # Update with momentum
        self.weight_momentum = momentum * self.weight_momentum + learning_rate * weights_gradient
        self.bias_momentum = momentum * self.bias_momentum + learning_rate * bias_gradient
        
        self.weights -= self.weight_momentum
        self.bias -= self.bias_momentum
        
        return input_gradient


# A max pooling layer: each neuron selects the maximum value in a local region (pooling size) of the input
class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size=2, stride=2):
        """
        Initialize MaxPooling2D layer
        Arguments:
            input_shape: tuple (channels, height, width)
            pool_size: int or tuple of two ints
            stride: int or tuple of two ints
        """
        self.channels, self.height, self.width = input_shape
        self.pool_size = (pool_size, pool_size) if isinstance(pool_size, int) else pool_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        
        # Calculate output dimensions
        self.out_height = (self.height - self.pool_size[0]) // self.stride[0] + 1
        self.out_width = (self.width - self.pool_size[1]) // self.stride[1] + 1
        self.output_shape = (self.channels, self.out_height, self.out_width)
        
        super().__init__(input_shape, self.output_shape)
        self.trainable = False

    def forward(self, input):
        """
        Forward pass of max pooling
        Arguments:
            input: Array of shape (batch_size, channels, height, width)
        Returns:
            output: Array of shape (batch_size, channels, out_height, out_width)
        """
        batch_size = input.shape[0]
        self.input = input
        
        output = np.zeros((batch_size, self.channels, self.out_height, self.out_width))
        self.max_indices = np.zeros((batch_size, self.channels, self.out_height, self.out_width, 2), dtype=np.int32)
        
        for b in range(batch_size):
            for c in range(self.channels):
                for i in range(self.out_height):
                    for j in range(self.out_width):
                        h_start = i * self.stride[0]
                        h_end = h_start + self.pool_size[0]
                        w_start = j * self.stride[1]
                        w_end = w_start + self.pool_size[1]
                        
                        window = input[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(window)
                        
                        # Store indices for backward pass
                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        self.max_indices[b, c, i, j] = (
                            h_start + max_idx[0],
                            w_start + max_idx[1]
                        )
        
        return output

    def backward(self, output_gradient, learning_rate=None):
        """
        Backward pass of the max pooling layer.
        Arguments:
            output_gradient: Array of shape (batch_size, channels, out_height, out_width)
            learning_rate: Not used (no parameters to update)
        Returns:
            input_gradient: Array of shape (batch_size, channels, height, width)
        """
        if self.input is None:
            raise RuntimeError("Backward pass called without a forward pass first")
            
        input_gradient = np.zeros_like(self.input)
        
        for b in range(self.input.shape[0]):
            for c in range(self.channels):
                for i in range(self.out_height):
                    for j in range(self.out_width):
                        # Get the stored 2D indices
                        h_idx, w_idx = self.max_indices[b, c, i, j]
                        # Add gradient to the max value's position
                        input_gradient[b, c, h_idx, w_idx] += output_gradient[b, c, i, j]
        
        return input_gradient


# A flattening layer: each neuron flattens the input into a 1D array
class FlattenLayer(Layer):
    def __init__(self, input_shape):
        output_shape = (np.prod(input_shape),)
        super().__init__(input_shape, output_shape)
        self.trainable = False  # Flatten has no parameters

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient.reshape(self.input_shape)


# A batch normalization layer: each neuron normalises the input to have a mean of 0 and a variance of 1
class BatchNormalization(Layer):
    def __init__(self, input_shape, epsilon=1e-5):
        """
        Initialisation of the batch normalization layer.
        Arguments:
            input_shape: tuple - shape of the input
            epsilon: float - small constant for numerical stability
        """
        super().__init__(input_shape, input_shape)
        self.epsilon = epsilon
        
        # Learnable parameters
        self.gamma = np.ones((1, *input_shape))
        self.beta = np.zeros((1, *input_shape))
        
        # Moving statistics for inference
        self.moving_mean = np.zeros((1, *input_shape))
        self.moving_var = np.ones((1, *input_shape))
        
        # Momentum terms
        self.gamma_momentum = np.zeros_like(self.gamma)
        self.beta_momentum = np.zeros_like(self.beta)
        
    def forward(self, input, training=True):
        """
        Forward pass of the batch normalization layer.
        Arguments:
            input: (batch_size, input_shape)
            training: bool - whether in training mode or inference mode
        Returns:
            output: (batch_size, input_shape)
        """
        self.input = input
        if training:
            mean = np.mean(input, axis=0, keepdims=True)
            var = np.var(input, axis=0, keepdims=True)
            
            # Update moving statistics with momentum (0.9 is common practice)
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_var = 0.9 * self.moving_var + 0.1 * var
        else:
            mean = self.moving_mean
            var = self.moving_var
        
        # Normalize
        self.normalised = (input - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.normalised + self.beta
    
    def backward(self, output_gradient, learning_rate, momentum=0.9):
        """
        Backward pass of the batch normalization layer.
        Arguments:
            output_gradient: gradient of the loss with respect to the output
            learning_rate: float - learning rate for parameter updates
            momentum: float - coefficient for the momentum updates
        Returns:
            input_gradient: gradient of the loss with respect to the input
        """
        N, C, H, W = self.input.shape
        std_inv = 1. / np.sqrt(self.moving_var + self.epsilon)
        
        # Compute gradients
        d_normalized = output_gradient * self.gamma
        d_var = np.sum(d_normalized * (self.input - self.moving_mean) * -0.5 * std_inv**3, axis=0, keepdims=True)
        d_mean = np.sum(d_normalized * -std_inv, axis=0, keepdims=True) + d_var * np.mean(-2. * (self.input - self.moving_mean), axis=0, keepdims=True)
        d_input = (d_normalized * std_inv) + (d_var * 2 * (self.input - self.moving_mean) / N) + (d_mean / N)
        
        # Compute parameter gradients
        d_gamma = np.sum(output_gradient * self.normalized, axis=(0, 2, 3), keepdims=True)
        d_beta = np.sum(output_gradient, axis=(0, 2, 3), keepdims=True)
        
        # Update parameters with momentum
        self.gamma_momentum = momentum * self.gamma_momentum + learning_rate * d_gamma
        self.beta_momentum = momentum * self.beta_momentum + learning_rate * d_beta
        
        self.gamma -= self.gamma_momentum
        self.beta -= self.beta_momentum
        
        return d_input


# A ReLU activation layer: each neuron applies a ReLU activation function to its input: max(0, x)
class ReLU(Layer):
    def __init__(self, input_shape):
        super().__init__(input_shape, input_shape)
        self.trainable = False  # ReLU has no parameters

    def forward(self, input):
        self.mask = (input > 0)
        return input * self.mask
    
    def backward(self, dout):
        return dout * self.mask
