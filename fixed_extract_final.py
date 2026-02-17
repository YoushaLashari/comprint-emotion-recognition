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
print("Model expects input: 48x48x1")
print("Model outputs: 48x48x1 (ComPrint noise residual)\n")

def extract_features(image_path):
    """Extract ComPrint features from an image"""
    try:
        # Read and preprocess image
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        # Convert to grayscale and resize to 48x48 (model requirement!)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (48, 48))  # CORRECT SIZE!
        img = img.astype(np.float32)
        
        # Shape: (48, 48) -> (1, 48, 48, 1)
        img = img[np.newaxis, :, :, np.newaxis]
        
        # Convert to tensor
        img_tensor = tf.convert_to_tensor(img)
        
        # Extract features
        output = infer(input_1=img_tensor)
        features = output['output_1'].numpy()[0]
        
        return features
        
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process CK+ dataset
print("Processing CK+ dataset...")
ck_path = Path("datasets/CK+")
ck_images = list(ck_path.rglob("*.png")) + list(ck_path.rglob("*.jpg"))
print(f"Found {len(ck_images)} images")

# Test with first image
print("\nTesting with first image...")
test_features = extract_features(ck_images[0])
if test_features is not None:
    print(f"✓ Success! Feature shape: {test_features.shape}")
    print(f"  Min: {test_features.min():.2f}, Max: {test_features.max():.2f}")
else:
    print("✗ Failed to extract features")
    exit()

print("\n✓ Feature extraction working! Processing all images...")

# Process all CK+ images
ck_features = {}
for img_path in tqdm(ck_images, desc="CK+"):
    features = extract_features(img_path)
    if features is not None:
        ck_features[str(img_path)] = features

# Save CK+ features
os.makedirs("output", exist_ok=True)
with open("output/ck_features_corrected.pkl", "wb") as f:
    pickle.dump(ck_features, f)
print(f"✓ CK+ done! Saved {len(ck_features)} features")

# Process JAFFE dataset
print("\nProcessing JAFFE dataset...")
jaffe_path = Path("datasets/JAFFE")
jaffe_images = list(jaffe_path.rglob("*.tiff")) + list(jaffe_path.rglob("*.jpg")) + list(jaffe_path.rglob("*.png"))
print(f"Found {len(jaffe_images)} images")

jaffe_features = {}
for img_path in tqdm(jaffe_images, desc="JAFFE"):
    features = extract_features(img_path)
    if features is not None:
        jaffe_features[str(img_path)] = features

# Save JAFFE features
with open("output/jaffe_features_corrected.pkl", "wb") as f:
    pickle.dump(jaffe_features, f)
print(f"✓ JAFFE done! Saved {len(jaffe_features)} features")

# Summary
print("\n" + "="*60)
print("CORRECTED EXTRACTION COMPLETE")
print("="*60)
print(f"CK+ Dataset: {len(ck_features)}/{len(ck_images)} images")
print(f"JAFFE Dataset: {len(jaffe_features)}/{len(jaffe_images)} images")
print(f"\nOutput files (CORRECTED):")
print(f"  - output/ck_features_corrected.pkl")
print(f"  - output/jaffe_features_corrected.pkl")
print(f"\nFeature size: 48x48x1 (correct model input/output)")