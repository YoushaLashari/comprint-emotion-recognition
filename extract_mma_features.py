import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm
import pickle

print("Loading ComPrint model...")
model = tf.saved_model.load("comprint/models/Comprint_Siamese_Full_jpg_ps_full")
infer = model.signatures["serving_default"]
print("Model loaded successfully!")
print("="*60)

def extract_features(image_path):
    """Extract ComPrint features from an image"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))
        img = img.astype(np.float32)
        img = img[np.newaxis, :, :, np.newaxis]
        
        img_tensor = tf.convert_to_tensor(img)
        output = infer(input_1=img_tensor)
        features = output['output_1'].numpy()[0]
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Check if MMA dataset exists
mma_path = Path("datasets/MMA FACIAL EXPRESSION")

if not mma_path.exists():
    mma_path = Path("datasets/MMA")

if not mma_path.exists():
    print("ERROR: MMA dataset not found!")
    print("Please check the folder name in datasets/")
    exit()

print(f"Found MMA dataset at: {mma_path}")
print("="*60)

# Find all images in train, test, valid folders
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
all_images = []

for split in ['train', 'test', 'valid']:
    split_path = mma_path / split
    if split_path.exists():
        for ext in image_extensions:
            all_images.extend(list(split_path.rglob(f'*{ext}')))
            all_images.extend(list(split_path.rglob(f'*{ext.upper()}')))

print(f"\nFound {len(all_images)} images in MMA dataset")

if len(all_images) == 0:
    print("ERROR: No images found! Check the folder structure.")
    exit()

# Test with first image
print("\nTesting with first image...")
test_features = extract_features(all_images[0])
if test_features is not None:
    print(f"✓ Success! Feature shape: {test_features.shape}")
else:
    print("✗ Failed to extract features")
    exit()

# Process all MMA images
print("\n✓ Feature extraction working! Processing all images...")
print("="*60)

mma_features = {}
failed_count = 0

for img_path in tqdm(all_images, desc="Extracting MMA features"):
    features = extract_features(img_path)
    if features is not None:
        mma_features[str(img_path)] = features
    else:
        failed_count += 1

# Save features
os.makedirs("output", exist_ok=True)
with open("output/mma_features_corrected.pkl", "wb") as f:
    pickle.dump(mma_features, f)

print("\n" + "="*60)
print("MMA EXTRACTION COMPLETE!")
print("="*60)
print(f"Successfully processed: {len(mma_features)} images")
print(f"Failed: {failed_count} images")
print(f"Success rate: {len(mma_features)/len(all_images)*100:.1f}%")
print(f"\nOutput saved to: output/mma_features_corrected.pkl")
print("\nNext step: Run organize_by_class.py to organize by emotions!")