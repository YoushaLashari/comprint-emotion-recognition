import pickle
import numpy as np
import cv2
import os
from pathlib import Path
import shutil

print("Reorganizing ComPrint features by emotion class...")
print("="*60)

def save_feature_map_image(feature_map, output_path):
    """Save ComPrint feature as pixelated visualization"""
    if len(feature_map.shape) == 3:
        feature_map = feature_map[:, :, 0]
    
    feature_highpass = cv2.Laplacian(feature_map, cv2.CV_32F)
    feature_highpass = np.abs(feature_highpass)
    
    if feature_highpass.max() > 0:
        feature_highpass = (feature_highpass / feature_highpass.max() * 255).astype(np.uint8)
    else:
        feature_highpass = np.zeros_like(feature_highpass, dtype=np.uint8)
    
    feature_pixelated = cv2.resize(feature_highpass, (512, 512), interpolation=cv2.INTER_NEAREST)
    colored = cv2.applyColorMap(feature_pixelated, cv2.COLORMAP_SUMMER)
    cv2.imwrite(str(output_path), colored)

def get_emotion_from_path(img_path, dataset_name):
    """Extract emotion class from image path"""
    path_parts = Path(img_path).parts
    
    if dataset_name == "CK+":
        # CK+ structure: datasets/CK+/anger/S010_004_00000017.png
        for part in path_parts:
            if part.lower() in ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise']:
                return part.lower()
    
    elif dataset_name == "JAFFE":
        # JAFFE structure: datasets/JAFFE/jaffe/KA.AN1.39.tiff
        # Filename encodes emotion: KA.AN1 = Angry, KA.DI1 = Disgust, etc.
        filename = Path(img_path).stem
        if '.AN' in filename or filename.startswith('AN'):
            return 'anger'
        elif '.DI' in filename or filename.startswith('DI'):
            return 'disgust'
        elif '.FE' in filename or filename.startswith('FE'):
            return 'fear'
        elif '.HA' in filename or filename.startswith('HA'):
            return 'happy'
        elif '.SA' in filename or filename.startswith('SA'):
            return 'sadness'
        elif '.SU' in filename or filename.startswith('SU'):
            return 'surprise'
        elif '.NE' in filename or filename.startswith('NE'):
            return 'neutral'
    
    elif dataset_name == "MMA":
        # MMA structure: datasets/MMA/train/angry/image.jpg
        for part in path_parts:
            if part.lower() in ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
                return part.lower()
    
    return 'unknown'

# Process CK+ Dataset
print("\n1. Processing CK+ Dataset...")
print("-" * 60)

if os.path.exists("output/ck_features_corrected.pkl"):
    with open("output/ck_features_corrected.pkl", "rb") as f:
        ck_features = pickle.load(f)
    
    emotion_counts = {}
    
    for img_path, features in ck_features.items():
        emotion = get_emotion_from_path(img_path, "CK+")
        
        # Create emotion folder
        emotion_dir = Path(f"output/visualizations_by_class/CK+/{emotion}")
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = Path(img_path).stem + "_comprint.png"
        output_path = emotion_dir / filename
        
        # Save visualization
        save_feature_map_image(features, output_path)
        
        # Count
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"✓ CK+ organized by emotion:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count} images")

else:
    print("✗ CK+ features file not found!")

# Process JAFFE Dataset
print("\n2. Processing JAFFE Dataset...")
print("-" * 60)

if os.path.exists("output/jaffe_features_corrected.pkl"):
    with open("output/jaffe_features_corrected.pkl", "rb") as f:
        jaffe_features = pickle.load(f)
    
    emotion_counts = {}
    
    for img_path, features in jaffe_features.items():
        emotion = get_emotion_from_path(img_path, "JAFFE")
        
        # Create emotion folder
        emotion_dir = Path(f"output/visualizations_by_class/JAFFE/{emotion}")
        emotion_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename
        filename = Path(img_path).stem + "_comprint.png"
        output_path = emotion_dir / filename
        
        # Save visualization
        save_feature_map_image(features, output_path)
        
        # Count
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    print(f"✓ JAFFE organized by emotion:")
    for emotion, count in sorted(emotion_counts.items()):
        print(f"   {emotion}: {count} images")

else:
    print("✗ JAFFE features file not found!")

# Process MMA Dataset (if available)
print("\n3. Checking for MMA Dataset...")
print("-" * 60)

# Check for MMA features file (not dataset folder)
if os.path.exists("output/mma_features_corrected.pkl"):
    print("✓ MMA features found! Processing...")
    
    if True:  # Always process if file exists
        with open("output/mma_features_corrected.pkl", "rb") as f:
            mma_features = pickle.load(f)
        
        emotion_counts = {'train': {}, 'test': {}, 'valid': {}}
        
        for img_path, features in mma_features.items():
            path_parts = Path(img_path).parts
            
            # Determine split (train/test/valid)
            split = None
            for part in path_parts:
                if part in ['train', 'test', 'valid']:
                    split = part
                    break
            
            if split is None:
                continue
            
            emotion = get_emotion_from_path(img_path, "MMA")
            
            # Create folder structure: MMA/train/angry/, MMA/test/angry/, etc.
            emotion_dir = Path(f"output/visualizations_by_class/MMA/{split}/{emotion}")
            emotion_dir.mkdir(parents=True, exist_ok=True)
            
            filename = Path(img_path).stem + "_comprint.png"
            output_path = emotion_dir / filename
            
            save_feature_map_image(features, output_path)
            
            emotion_counts[split][emotion] = emotion_counts[split].get(emotion, 0) + 1
        
        print(f"✓ MMA organized by split and emotion:")
        for split in ['train', 'test', 'valid']:
            print(f"\n   {split.upper()}:")
            for emotion, count in sorted(emotion_counts[split].items()):
                print(f"      {emotion}: {count} images")
else:
    print("⏸️  MMA dataset not found in datasets folder")
    print("   Add it later and run this script again")

print("\n" + "="*60)
print("ORGANIZATION COMPLETE!")
print("="*60)
print(f"\nOutput structure:")
print(f"output/visualizations_by_class/")
print(f"├── CK+/")
print(f"│   ├── anger/")
print(f"│   ├── disgust/")
print(f"│   └── ... (all emotion classes)")
print(f"├── JAFFE/")
print(f"│   ├── anger/")
print(f"│   ├── happy/")
print(f"│   └── ... (all emotion classes)")
print(f"└── MMA/ (if available)")
print(f"    ├── train/")
print(f"    │   ├── angry/")
print(f"    │   └── ...")
print(f"    ├── test/")
print(f"    └── valid/")
print(f"\n✓ Now each emotion class has its own folder!")
print(f"✓ Easy to use for training machine learning models!")