import numpy as np
from NeuralNetwork.AbstractClasses import Loss, Optimiser

# A loss function for the IoU (Intersection over Union)
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