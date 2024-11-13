import numpy as np
import os
from PIL import Image

class DataSetForBBD: # Bounding Box Detection
    """
    This a class that converts images and labels into understandable format for the model. (numbers)
    This class assumes that the directories are organised in the following way:
    base_dir/
        train/
            images/
            labels/
        test/
            images/
            labels/
        val/
            images/
            labels/
    It also assumes that labels are stored in txt files with the same name as the images.
    It resizes the images to 224x224. Keeps images in RGB format.
    """
    def __init__(self, base_dir, batch_size=32, shuffle=True, seed=42):
        self.base_dir = base_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        self.dataset_paths = {
            'train': {
                'images': [],
                'labels': []
            },
            'test': {
                'images': [],
                'labels': []
            },
            'val': {
                'images': [],
                'labels': []
            }
        }

        self.batch_counts = {
            'train': 0,
            'test': 0,
            'val': 0
        }

        if not os.path.exists(base_dir):
            raise FileNotFoundError(f"The base directory {base_dir} does not exist")
        self._load_data()
    
    def _load_data(self):
        """ Loads paths of images and labels from the directory structure into self.dataset_paths """
        for dataset_type in ['train', 'test', 'val']:
            img_dir = os.path.join(self.base_dir, dataset_type, 'images')
            label_dir = os.path.join(self.base_dir, dataset_type, 'labels')

            for img_file in os.listdir(img_dir):
                text_file_name = img_file.replace('.jpg', '.txt')
                if not os.path.exists(os.path.join(label_dir, text_file_name)):
                    print(f"The label file {text_file_name} does not exist")
                    continue
                img_path = os.path.join(img_dir, img_file)
                label_path = os.path.join(label_dir, text_file_name)
                self.dataset_paths[dataset_type]['images'].append(img_path)
                self.dataset_paths[dataset_type]['labels'].append(label_path)

            if self.shuffle:
                np.random.seed(self.seed)
                np.random.shuffle(self.dataset_paths[dataset_type]['images'])
                np.random.seed(self.seed)
                np.random.shuffle(self.dataset_paths[dataset_type]['labels'])
    
    
    def _preprocess_image(self, img_path):
        """ Opens image, resizes it to 224x224 and converts it to numpy array. """
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = img.convert('RGB')
        img = np.array(img)
        return img
    
    def _preprocess_label(self, label_path):
        """ Reads label file and returns list of bounding boxes. It assumes that the label file is in YOLO format (class_id, x_center, y_center, width, height). (normalized to 0-1) """
        with open(label_path, 'r') as f:
            coords = []
            for line in f:
                _, x_center, y_center, width, height = map(float, line.strip().split())
                coords.append([x_center, y_center, width, height]) # no need for class_id
        return coords
    
    def get_batch(self, dataset_type):
        """Returns a batch of images and labels."""
        images = np.zeros((self.batch_size, 3, 224, 224))
        labels = []
        n_of_files = len(self.dataset_paths[dataset_type]['images'])
        start_idx = (self.batch_counts[dataset_type] * self.batch_size) % n_of_files

        for i in range(self.batch_size):
            idx = (start_idx + i) % n_of_files
            
            img = self._preprocess_image(self.dataset_paths[dataset_type]['images'][idx])
            img = np.transpose(img, (2, 0, 1))
            boxes = self._preprocess_label(self.dataset_paths[dataset_type]['labels'][idx])
            
            images[i] = img
            labels.append(boxes)
        
        self.batch_counts[dataset_type] += 1
        
        max_boxes = max(len(boxes) for boxes in labels)
        padded_labels = np.zeros((self.batch_size, max_boxes, 4))
        for i, boxes in enumerate(labels):
            padded_labels[i, :len(boxes)] = boxes
        
        return images, padded_labels
    

dataset = DataSetForBBD('datasets/licence-plate-detection/', batch_size=64, shuffle=True, seed=42)