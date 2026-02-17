"""
CNN Training on ORIGINAL IMAGES
Converted from Colab notebook to work in VS Code
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

print("=" * 80)
print("CNN TRAINING - ORIGINAL IMAGES")
print("=" * 80)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Dataset path - MODIFY THIS to your actual path
dataset_path = 'datasets/CK+'  # Local path for VS Code

# Model parameters
image_height = 224
image_width = 224
batch_size = 32
epochs = 50

# Create output directory
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

print(f"\nDataset Path: {dataset_path}")
print(f"Image Size: {image_height}x{image_width}")
print(f"Batch Size: {batch_size}")
print(f"Epochs: {epochs}")

# ============================================================================
# DATA LOADING
# ============================================================================

print("\n" + "=" * 80)
print("LOADING DATA")
print("=" * 80)

# Count classes
num_classes = len([d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))])
classes = sorted([d for d in os.listdir(dataset_path) 
                  if os.path.isdir(os.path.join(dataset_path, d))])

print(f"\nNumber of Classes: {num_classes}")
print(f"Classes: {classes}")

# Count images per class
for class_name in classes:
    class_path = os.path.join(dataset_path, class_name)
    count = len([f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    print(f"  {class_name}: {count} images")

# ============================================================================
# DATA AUGMENTATION
# ============================================================================

print("\n" + "=" * 80)
print("SETTING UP DATA AUGMENTATION")
print("=" * 80)

# Data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 80% train, 20% validation
)

# Generate training dataset
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Generate validation dataset
validation_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(image_height, image_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

print(f"\n✓ Training samples: {train_generator.samples}")
print(f"✓ Validation samples: {validation_generator.samples}")

# ============================================================================
# BUILD CNN MODEL
# ============================================================================

print("\n" + "=" * 80)
print("BUILDING CNN MODEL")
print("=" * 80)

def build_cnn_model(input_shape, num_classes):
    """Build a simple CNN model"""
    model = models.Sequential([
        # Convolutional layers
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

# Create model
input_shape = (image_height, image_width, 3)
model = build_cnn_model(input_shape, num_classes)

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Architecture:")
model.summary()

# ============================================================================
# TRAIN MODEL
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING MODEL")
print("=" * 80)

# Define callbacks
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001,
    verbose=1
)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# Train the model
# For small datasets, don't limit steps - let it use all data
print(f"\nTraining for {epochs} epochs...")
print(f"Training batches per epoch: {len(train_generator)}")
print(f"Validation batches per epoch: {len(validation_generator)}")

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs,
    callbacks=[reduce_lr, early_stopping],
    verbose=1
)

# ============================================================================
# SAVE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("SAVING MODEL")
print("=" * 80)

model_path = os.path.join(output_dir, 'cnn_original_images.h5')
model.save(model_path)
print(f"✓ Model saved: {model_path}")

# ============================================================================
# EVALUATE MODEL
# ============================================================================

print("\n" + "=" * 80)
print("EVALUATING MODEL")
print("=" * 80)

# Evaluate on validation set
test_loss, test_accuracy = model.evaluate(validation_generator, verbose=0)
print(f"\n✓ Validation Accuracy: {test_accuracy * 100:.2f}%")
print(f"✓ Validation Loss: {test_loss:.4f}")

# ============================================================================
# GENERATE PREDICTIONS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING PREDICTIONS")
print("=" * 80)

# Generate predictions
validation_generator.reset()
predictions = model.predict(validation_generator, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

# True labels
true_classes = validation_generator.classes
class_labels = list(validation_generator.class_indices.keys())

# ============================================================================
# CONFUSION MATRIX
# ============================================================================

print("\n" + "=" * 80)
print("CONFUSION MATRIX")
print("=" * 80)

# Calculate confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_labels,
            yticklabels=class_labels)
plt.title('Confusion Matrix - Original Images CNN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()

cm_path = os.path.join(output_dir, 'confusion_matrix_original.png')
plt.savefig(cm_path, dpi=150)
print(f"✓ Confusion matrix saved: {cm_path}")
plt.close()

# ============================================================================
# CLASSIFICATION REPORT
# ============================================================================

print("\n" + "=" * 80)
print("CLASSIFICATION REPORT")
print("=" * 80)

report = classification_report(true_classes, predicted_classes, 
                               target_names=class_labels)
print("\n" + report)

# Save report to file
report_path = os.path.join(output_dir, 'classification_report_original.txt')
with open(report_path, 'w') as f:
    f.write("CNN Classification Report - Original Images\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Validation Accuracy: {test_accuracy * 100:.2f}%\n")
    f.write(f"Validation Loss: {test_loss:.4f}\n\n")
    f.write(report)
print(f"✓ Report saved: {report_path}")

# ============================================================================
# TRAINING HISTORY PLOTS
# ============================================================================

print("\n" + "=" * 80)
print("GENERATING TRAINING PLOTS")
print("=" * 80)

# Plot training history
plt.figure(figsize=(14, 5))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy - Original Images', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, alpha=0.3)

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss - Original Images', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
history_path = os.path.join(output_dir, 'training_history_original.png')
plt.savefig(history_path, dpi=150)
print(f"✓ Training plots saved: {history_path}")
plt.close()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("TRAINING COMPLETE!")
print("=" * 80)
print("\nSaved Files:")
print(f"  1. Model: {model_path}")
print(f"  2. Confusion Matrix: {cm_path}")
print(f"  3. Classification Report: {report_path}")
print(f"  4. Training History: {history_path}")
print("\n" + "=" * 80)
print(f"Final Validation Accuracy: {test_accuracy * 100:.2f}%")
print("=" * 80)