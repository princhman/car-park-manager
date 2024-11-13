import numpy as np
from NeuralNetwork.Layers import Conv2DLayer
from NeuralNetwork.AbstractClasses import Layer
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
    
    def forward(self, input):
        return self.conv.forward(input)
    
    def backward(self, output_gradient, learning_rate):
        return self.conv.backward(output_gradient, learning_rate)


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
        
        