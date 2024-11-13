from NeuralNetwork.Layers import *
from NeuralNetwork.Functions import *
from NeuralNetwork.Other import *
class NumberPlateDetector:
    def __init__(self, input_shape=(3, 224, 224), num_classes=4):
        self.model = Model()
        
        # Backbone Network (optimized for small object detection)
        # First block
        self.model.add(Conv2DLayer(input_shape, 32, kernel_size=3))  # Output: (32, 222, 222)
        self.model.add(BatchNormalization((32, 222, 222)))
        self.model.add(ReLU((32, 222, 222)))
        self.model.add(MaxPooling2D((32, 222, 222), pool_size=2, stride=2))  # Output: (32, 111, 111)
        
        # Second block
        self.model.add(Conv2DLayer((32, 111, 111), 64, kernel_size=3))  # Output: (64, 109, 109)
        self.model.add(BatchNormalization((64, 109, 109)))
        self.model.add(ReLU((64, 109, 109)))
        self.model.add(MaxPooling2D((64, 109, 109), pool_size=2, stride=2))  # Output: (64, 54, 54)
        
        # Third block (enhanced for small object detection)
        self.model.add(Conv2DLayer((64, 54, 54), 128, kernel_size=3))  # Output: (128, 52, 52)
        self.model.add(BatchNormalization((128, 52, 52)))
        self.model.add(ReLU((128, 52, 52)))
        self.model.add(Conv2DLayer((128, 52, 52), 64, kernel_size=1))    # Output: (64, 52, 52)
        self.model.add(BatchNormalization((64, 52, 52)))
        self.model.add(ReLU((64, 52, 52)))
        self.model.add(Conv2DLayer((64, 52, 52), 128, kernel_size=3))   # Output: (128, 50, 50)
        self.model.add(BatchNormalization((128, 50, 50)))
        self.model.add(ReLU((128, 50, 50)))
        self.model.add(MaxPooling2D((128, 50, 50), pool_size=2, stride=2))  # Output: (128, 25, 25)
        
        # Fourth block
        self.model.add(Conv2DLayer((128, 25, 25), 256, kernel_size=3))   # Output: (256, 23, 23)
        self.model.add(BatchNormalization((256, 23, 23)))
        self.model.add(ReLU((256, 23, 23)))
        self.model.add(Conv2DLayer((256, 23, 23), 128, kernel_size=1))    # Output: (128, 23, 23)
        self.model.add(BatchNormalization((128, 23, 23)))
        self.model.add(ReLU((128, 23, 23)))
        self.model.add(Conv2DLayer((128, 23, 23), 256, kernel_size=3))    # Output: (256, 21, 21)
        self.model.add(BatchNormalization((256, 21, 21)))
        self.model.add(ReLU((256, 21, 21)))
        
        # Detection head
        num_anchors = 3
        output_channels = num_anchors * (5 + num_classes)  # 5 for bbox coords + confidence
        self.detection_head = DetectionHead((256, 21, 21), num_classes, num_anchors)
        self.model.add(self.detection_head)
        
        # Anchor boxes optimized for number plates (typically rectangular)
        self.anchors = AnchorBox(
            scales=[16, 32, 64],     # Smaller scales for number plates
            ratios=[0.25, 0.3, 0.4]  # More rectangular ratios typical for plates
        )
        
        # Initialize loss and optimizer
        self.loss = IoULoss()
        self.optimizer = Adam()

    def train(self, x, y):
        """
        x: shape (batch_size, 3, 224, 224)
        y: shape (batch_size, max_boxes, 4)
        """
        self.model.train()
        predictions = self.model.forward(x)
        
        # Ensure predictions match the expected shape
        batch_size = x.shape[0]
        grid_size = predictions.shape[2:4]
        predictions = predictions.reshape(batch_size, 3, -1, grid_size[0], grid_size[1])
        
        # Calculate loss only for valid boxes (non-zero)
        valid_mask = np.any(y != 0, axis=2)
        loss = self.loss.calculate(predictions, y[valid_mask])
        gradients = self.loss.gradient(predictions, y[valid_mask])
        
        # Backward pass through the model
        for layer in reversed(self.model.layers):
            if hasattr(layer, 'backward'):
                gradients = layer.backward(gradients, self.optimizer.learning_rate)
        
        self.optimizer.step(self.model.layers)
        return loss

    def predict(self, x):
        """
        x: shape (batch_size, 3, 224, 224)
        """
        if x.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected input shape (batch_size, 3, 224, 224), got {x.shape}")
        
        self.model.eval()
        predictions = self.model.forward(x)
        return self._post_process(predictions)

    def _post_process(self, predictions, conf_threshold=0.4, iou_threshold=0.4):  # Lower thresholds for plates
        """Post-process predictions to get final number plate bounding boxes"""
        batch_size = predictions.shape[0]
        grid_size = predictions.shape[2:4]
        
        # Reshape predictions
        predictions = predictions.reshape(batch_size, 3, -1, grid_size[0], grid_size[1])
        
        # Extract components
        confidence = predictions[:, :, 0, :, :]
        boxes = predictions[:, :, 1:5, :, :]  # x, y, w, h
        
        # Filter by confidence
        mask = confidence > conf_threshold
        filtered_boxes = boxes[mask]
        filtered_scores = confidence[mask]
        
        # Apply NMS
        keep = self._nms(filtered_boxes, filtered_scores, iou_threshold)
        
        return filtered_boxes[keep], filtered_scores[keep]

    def _nms(self, boxes, scores, iou_threshold): # Non-maximum suppression
        indices = np.argsort(scores)[::-1]
        keep = []
        
        while indices.size > 0:
            i = indices[0]
            keep.append(i)
            
            if indices.size == 1:
                break
                
            iou = self._compute_iou(boxes[i], boxes[indices[1:]])
            mask = iou <= iou_threshold
            indices = indices[1:][mask]
            
        return keep

    def _compute_iou(self, box, boxes): # It is used in the NMS funciton (non-maximum supression)
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection
        
        return intersection / (union + 1e-6)

    def _validate_data(self, x, y=None):
        """Validate input data formats"""
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (batch_size, channels, height, width), got {x.ndim}D")
        if x.shape[1:] != (3, 224, 224):
            raise ValueError(f"Expected shape (batch_size, 3, 224, 224), got {x.shape}")
        
        if y is not None:
            if y.ndim != 3:
                raise ValueError(f"Expected 3D labels (batch_size, max_boxes, 4), got {y.ndim}D")
            if y.shape[2] != 4:
                raise ValueError(f"Expected 4 values per box (x,y,w,h), got {y.shape[2]}")

