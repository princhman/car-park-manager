import numpy as np
from Datasets import dataset
from ML_models import NumberPlateDetector

# Initialize model
model = NumberPlateDetector()

# Training hyperparameters
num_epochs = 50
best_loss = float('inf')
patience = 5  # For early stopping
patience_counter = 0

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0
    num_batches = 0
    print(f"Epoch {epoch+1} started")
    # Training phase
    while True:
        try:
            # Get batch of training data
            images, labels = dataset.get_batch('train')
            
            # Convert images to the expected format (B, C, H, W)
            images = np.transpose(images, (0, 3, 1, 2))
            
            # Debug: Print input shape
            print(f"Batch {num_batches+1} input shape: {images.shape}")
            
            # Train on batch
            batch_loss = model.train(images, labels)
            epoch_loss += batch_loss
            num_batches += 1
            
            # Print batch progress
            if num_batches % 10 == 0:
                print(f"Epoch {epoch+1}, Batch {num_batches}, Loss: {batch_loss:.4f}")
                
        except StopIteration:
            break
    
    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    print(f"Epoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
    
    # Validation phase
    val_loss = 0
    val_batches = 0
    
    while True:
        try:
            # Get batch of validation data
            val_images, val_labels = dataset.get_batch('val')
            val_images = np.transpose(val_images, (0, 3, 1, 2))
            
            # Debug: Print validation input shape
            print(f"Validation Batch {val_batches+1} input shape: {val_images.shape}")
            
            # Get predictions
            predictions = model.predict(val_images)
            # Calculate validation loss
            val_batch_loss = model.loss.calculate(predictions, val_labels)
            val_loss += val_batch_loss
            val_batches += 1
            
        except StopIteration:
            break
    
    avg_val_loss = val_loss / val_batches
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    # Early stopping check
    if avg_val_loss < best_loss:
        best_loss = avg_val_loss
        patience_counter = 0
        # Here you could save the model if needed
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered")
            break

print("Training completed!")
