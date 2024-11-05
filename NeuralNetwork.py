import numpy as np
from abc import ABC, abstractmethod

# A template for other layers: default layer
class Layer(ABC):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape 
        self.output_shape = output_shape
        self.trainable = True  
    
    @abstractmethod
    def forward(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backward(self, output_gradient, learning_rate):
        raise NotImplementedError

# An optimiser template: each optimiser updates the parameters of the layers
class Optimiser(ABC):    
    @abstractmethod
    def step(self, layers):
        raise NotImplementedError

# A loss template: each loss function calculates the loss and its gradient
class Loss(ABC):    
    @abstractmethod
    def calculate(self, output, target):
        raise NotImplementedError
    
    @abstractmethod
    def gradient(self, output, target):
        raise NotImplementedError

# A fully connected layer: each neuron connects to all neurons in the next layer
class FullyConnectedLayer(Layer):
    def __init__(self, input_shape, output_shape):
        super().__init__(input_shape, output_shape)
        # Xavier/Glorot initialization and bias initialisation
        scale = np.sqrt(2.0 / (input_shape[0] + output_shape[0]))
        self.weights = np.random.randn(output_shape[0], input_shape[0]) * scale
        self.bias = np.zeros((output_shape[0], 1)) 
        
        # momentum terms are used to optimise the training process
        self.weight_momentum = np.zeros_like(self.weights)
        self.bias_momentum = np.zeros_like(self.bias)
        
    def forward(self, input):
        self.input = input  # cache input for backward pass
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient, learning_rate, momentum=0.9):
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        self.weight_momentum = momentum * self.weight_momentum + (1 - momentum) * weights_gradient
        self.bias_momentum = momentum * self.bias_momentum + (1 - momentum) * output_gradient
        return input_gradient

# An anchor box class: purpose is to generate anchor boxes for each position in the feature map
class AnchorBox:
    def __init__(self, scales, ratios):
        self.scales = scales
        self.ratios = ratios
        
    def generate_anchors(self, feature_map_size):
        anchors = []
        for y in range(feature_map_size[0]):
            for x in range(feature_map_size[1]):
                for scale in self.scales:
                    for ratio in self.ratios:
                        w = scale * np.sqrt(ratio)
                        h = scale / np.sqrt(ratio)
                        anchors.append([x, y, w, h])
        return np.array(anchors)

# A detection head class: purpose is to predict the bounding boxes of the objects in the image
class DetectionHead(Layer):
    def __init__(self, input_shape, num_classes, num_anchors):
        # For each anchor: [x, y, w, h, objectness, num_classes]
        output_shape = (input_shape[0], num_anchors * (5 + num_classes))
        super().__init__(input_shape, output_shape)
        self.conv = Conv2DLayer(input_shape, output_shape[1], 3)

# A 2D convolutional layer: each neuron connects to a local region (kernel) in the input
class Conv2DLayer(Layer):
    def __init__(self, input_shape, num_filters, kernel_size, stride=1, padding=0):
        # Calculate output shape
        batch_size, channels, height, width = input_shape
        out_height = ((height + 2*padding - kernel_size) // stride) + 1
        out_width = ((width + 2*padding - kernel_size) // stride) + 1
        output_shape = (batch_size, num_filters, out_height, out_width)
        
        super().__init__(input_shape, output_shape)
        
        # He initialization
        scale = np.sqrt(2.0 / (channels * kernel_size * kernel_size))
        self.weights = np.random.randn(num_filters, channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros((num_filters, 1))
        
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

    def forward(self, input):
        output = np.zeros((self.output_shape[0], input.shape[1], input.shape[2])) # empty output array, with required dimensions
        # iteration over each output feature map, extracting the relevant patch (kernel) from the input and calculating the output
        for i in range(self.output_shape[0]):
            for j in range(input.shape[1] - self.kernel_size + 1):
                for k in range(input.shape[2] - self.kernel_size + 1):
                    patch = input[:, j:j+self.kernel_size, k:k+self.kernel_size]
                    output[i, j, k] = np.sum(patch * self.weights[i]) + self.bias[i] # output value calcuation
        return output
    
    def backward(self, input, output_gradient, learning_rate):
        input_gradient = np.zeros_like(input)
        weights_gradient = np.zeros_like(self.weights)
        bias_gradient = np.zeros_like(self.bias)
        
        # calculating the gradients
        for i in range(self.output_shape[0]):
            for j in range(input.shape[1] - self.kernel_size + 1):
                for k in range(input.shape[2] - self.kernel_size + 1):
                    patch = input[:, j:j+self.kernel_size, k:k+self.kernel_size]
                    output_grad_patch = output_gradient[i, j, k]
                    input_gradient[:, j:j+self.kernel_size, k:k+self.kernel_size] += self.weights[i] * output_grad_patch
                    weights_gradient[i] += patch * output_grad_patch
                    bias_gradient[i] += output_grad_patch
        
        # updating the weights and bias
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient
        
        return input_gradient

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

# A max pooling layer: each neuron selects the maximum value in a local region (pooling size) of the input
class MaxPooling2D(Layer):
    def __init__(self, input_shape, pool_size=2, stride=None):
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        
        batch_size, channels, height, width = input_shape
        out_height = (height - pool_size) // self.stride + 1
        out_width = (width - pool_size) // self.stride + 1
        output_shape = (batch_size, channels, out_height, out_width)
        
        super().__init__(input_shape, output_shape)
        self.trainable = False  # MaxPool has no parameters
    
    def forward(self, input):
        output = np.zeros((self.output_shape[0], input.shape[1] // self.pool_size, input.shape[2] // self.pool_size))
        for i in range(self.output_shape[0]):
            for j in range(output.shape[1]):
                for k in range(output.shape[2]):
                    output[i, j, k] = np.max(input[i, j*self.pool_size:(j+1)*self.pool_size, k*self.pool_size:(k+1)*self.pool_size])
        return output
    
    def backward(self, input, output_gradient):
        input_gradient = np.zeros_like(input)
        for i in range(self.output_shape[0]):
            for j in range(input.shape[1] // self.pool_size):
                for k in range(input.shape[2] // self.pool_size):
                    input_gradient[i, j*self.pool_size:(j+1)*self.pool_size, k*self.pool_size:(k+1)*self.pool_size] = output_gradient[i, j, k]
        return input_gradient

# A flattening layer: each neuron flattens the input into a 1D array
class FlattenLayer(Layer):
    def __init__(self, input_shape):
        super().__init__(input_shape, np.prod(input_shape, 1))
        self.trainable = False  # Flatten has no parameters

    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient.reshape(self.input_shape)

# A batch normalization layer: each neuron normalises the input to have a mean of 0 and a variance of 1
class BatchNormalization(Layer):
    def __init__(self, input_shape, epsilon=1e-5):
        super().__init__(input_shape, input_shape)
        self.epsilon = epsilon
        self.gamma = np.ones(input_shape)
        self.beta = np.zeros(input_shape)
        self.moving_mean = np.zeros(input_shape)
        self.moving_var = np.ones(input_shape)
        
    def forward(self, input, training=True):
        if training:
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            
            # Update moving statistics
            self.moving_mean = 0.9 * self.moving_mean + 0.1 * mean
            self.moving_var = 0.9 * self.moving_var + 0.1 * var
        else:
            mean = self.moving_mean
            var = self.moving_var
            
        # Normalize
        self.input = input
        self.normalized = (input - mean) / np.sqrt(var + self.epsilon)
        return self.gamma * self.normalized + self.beta
    
    def backward(self, output_gradient, learning_rate):
        # Gradients for gamma and beta
        gamma_gradient = np.sum(output_gradient * self.normalized, axis=0)
        beta_gradient = np.sum(output_gradient, axis=0)
        
        # Update parameters
        self.gamma -= learning_rate * gamma_gradient
        self.beta -= learning_rate * beta_gradient
        
        # Input gradient
        return output_gradient * self.gamma / np.sqrt(self.var + self.epsilon)

# A dropout layer: each neuron randomly drops out with a probability of drop_rate
class Dropout(Layer):
    def __init__(self, input_shape, drop_rate=0.5):
        super().__init__(input_shape, input_shape)
        self.drop_rate = drop_rate
        self.trainable = False
        
    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(1, 1-self.drop_rate, input.shape) / (1-self.drop_rate)
            return input * self.mask
        return input
    
    def backward(self, output_gradient, learning_rate=None):
        return output_gradient * self.mask

# A IoU loss function: each neuron calculates the IoU (Intersection over Union) between the predicted and target bounding boxes
import numpy as np

class IoULoss(Loss):
    @staticmethod
    def compute_iou(boxes1, boxes2):

        # Compute intersection
        x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        # Compute union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-6)

    def calculate(self, output, target):
        iou = self.compute_iou(output, target)
        return 1 - np.mean(iou)

    def gradient(self, output, target):
        # Convert to corner format
        pred_boxes = np.zeros_like(output)
        pred_boxes[:, 0] = output[:, 0] - output[:, 2] / 2  # x1
        pred_boxes[:, 1] = output[:, 1] - output[:, 3] / 2  # y1
        pred_boxes[:, 2] = output[:, 0] + output[:, 2] / 2  # x2
        pred_boxes[:, 3] = output[:, 1] + output[:, 3] / 2  # y2
        
        true_boxes = np.zeros_like(target)
        true_boxes[:, 0] = target[:, 0] - target[:, 2] / 2
        true_boxes[:, 1] = target[:, 1] - target[:, 3] / 2
        true_boxes[:, 2] = target[:, 0] + target[:, 2] / 2
        true_boxes[:, 3] = target[:, 1] + target[:, 3] / 2
        
        # Compute intersection points
        ix1 = np.maximum(pred_boxes[:, 0], true_boxes[:, 0])
        iy1 = np.maximum(pred_boxes[:, 1], true_boxes[:, 1])
        ix2 = np.minimum(pred_boxes[:, 2], true_boxes[:, 2])
        iy2 = np.minimum(pred_boxes[:, 3], true_boxes[:, 3])
        
        # Compute gradients
        grad_x1 = -np.where(pred_boxes[:, 0] > true_boxes[:, 0], 1, 0)
        grad_y1 = -np.where(pred_boxes[:, 1] > true_boxes[:, 1], 1, 0)
        grad_x2 = np.where(pred_boxes[:, 2] < true_boxes[:, 2], 1, 0)
        grad_y2 = np.where(pred_boxes[:, 3] < true_boxes[:, 3], 1, 0)
        
        # Convert corner gradients to center format
        grad_x = (grad_x1 + grad_x2) / 2
        grad_y = (grad_y1 + grad_y2) / 2
        grad_w = (-grad_x1 + grad_x2) / 2
        grad_h = (-grad_y1 + grad_y2) / 2
        
        return np.stack([grad_x, grad_y, grad_w, grad_h], axis=1)

# An Adam optimiser: each neuron updates the parameters of the layers using the Adam algorithm
class Adam(Optimiser):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Decay rate for first moment
        self.beta2 = beta2  # Decay rate for second moment
        self.epsilon = epsilon
        self.m = {}  # First moment estimates
        self.v = {}  # Second moment estimates
        self.t = 0   # Timestep
    
    def step(self, layers):
        self.t += 1
        for layer in layers:
            if not hasattr(layer, 'weights') or not layer.trainable:
                continue
                
            # Initialize momentum if not exists
            if layer not in self.m:
                self.m[layer] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }
                self.v[layer] = {
                    'weights': np.zeros_like(layer.weights),
                    'bias': np.zeros_like(layer.bias)
                }
            
            # Update for weights
            self.m[layer]['weights'] = (self.beta1 * self.m[layer]['weights'] + 
                                      (1 - self.beta1) * layer.weight_gradient)
            self.v[layer]['weights'] = (self.beta2 * self.v[layer]['weights'] + 
                                      (1 - self.beta2) * np.square(layer.weight_gradient))
            
            # Bias correction
            m_hat = self.m[layer]['weights'] / (1 - self.beta1**self.t)
            v_hat = self.v[layer]['weights'] / (1 - self.beta2**self.t)
            
            # Update weights
            layer.weights -= (self.learning_rate * m_hat / 
                            (np.sqrt(v_hat) + self.epsilon))
            
            # Same process for bias
            self.m[layer]['bias'] = (self.beta1 * self.m[layer]['bias'] + 
                                   (1 - self.beta1) * layer.bias_gradient)
            self.v[layer]['bias'] = (self.beta2 * self.v[layer]['bias'] + 
                                   (1 - self.beta2) * np.square(layer.bias_gradient))
            
            m_hat = self.m[layer]['bias'] / (1 - self.beta1**self.t)
            v_hat = self.v[layer]['bias'] / (1 - self.beta2**self.t)
            
            layer.bias -= (self.learning_rate * m_hat / 
                         (np.sqrt(v_hat) + self.epsilon))

class Model:
    def __init__(self):
        self.layers = []
        self.training = True
    
    def add(self, layer):
        self.layers.append(layer)
    
    def forward(self, input):
        for layer in self.layers:
            if hasattr(layer, 'training'):
                input = layer.forward(input, training=self.training)
            else:
                input = layer.forward(input)
        return input
    
    def train(self):
        self.training = True
    
    def eval(self):
        self.training = False
        
        