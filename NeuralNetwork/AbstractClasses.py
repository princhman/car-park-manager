import numpy as np
from abc import ABC, abstractmethod

# A template for all layers
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