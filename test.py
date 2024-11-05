import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_yolo_label(label_path):
    """Read YOLO format label file and return list of bounding boxes"""
    print(f"Reading label file: {label_path}")
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.strip().split())
            boxes.append([x_center, y_center, width, height])
    print(f"Found {len(boxes)} boxes in label file")
    return boxes

def draw_boxes(img_path, label_path):
    """Draw bounding boxes on image from YOLO format labels"""
    print(f"\nProcessing image: {img_path}")
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    print("Successfully read image")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"Image dimensions: {img.shape}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    
    # Read and convert YOLO format boxes
    boxes = read_yolo_label(label_path)
    for box in boxes:
        x_center, y_center, w, h = box
        
        # Convert normalized coordinates to pixel coordinates
        x1 = int((x_center - w/2) * width)
        y1 = int((y_center - h/2) * height)
        x2 = int((x_center + w/2) * width)
        y2 = int((y_center + h/2) * height)
        
        # Draw rectangle
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return img

# Setup plotting
plt.figure(figsize=(12, 8))

# Plot more images from each set (train, test, val)
datasets = [
    ('train', 6),
    ('test', 6),
    ('val', 6)
]

plot_idx = 1
base_path = 'datasets/licence-plate-detection'

print("Starting script...")
print(f"Looking for images in: {base_path}")

for dataset_type, num_images in datasets:
    img_dir = os.path.join(base_path, dataset_type, 'images')
    label_dir = os.path.join(base_path, dataset_type, 'labels')
    
    print(f"\nProcessing {dataset_type} dataset")
    print(f"Image directory: {img_dir}")
    print(f"Label directory: {label_dir}")
    
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory does not exist: {img_dir}")
        continue
    if not os.path.exists(label_dir):
        print(f"Warning: Label directory does not exist: {label_dir}")
        continue
    
    image_files = sorted(os.listdir(img_dir))[:num_images]
    print(f"Found {len(image_files)} images to process")
    
    for img_file in image_files:
        # Get corresponding label file
        label_file = os.path.splitext(img_file)[0] + '.txt'
        label_path = os.path.join(label_dir, label_file)
        
        # Draw image with bounding boxes
        img_path = os.path.join(img_dir, img_file)
        img = draw_boxes(img_path, label_path)
        
        plt.subplot(len(datasets), num_images, plot_idx)
        plt.imshow(img)
        plt.title(f'{dataset_type}: {img_file}', fontsize=8)
        plt.axis('off')
        
        plot_idx += 1

plt.tight_layout(pad=1.5)
plt.show()