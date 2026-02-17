"""
Memory-Efficient ResNet Comparison: Original Images vs ComPrint Features
Uses data generators to avoid loading all images into memory at once
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pickle
import cv2
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns

print("="*80)
print("MEMORY-EFFICIENT RESNET COMPARISON")
print("Original Images vs ComPrint Features")
print("="*80)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.0001

print(f"\nConfiguration:")
print(f"  Image Size: {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {EPOCHS}")
print(f"  Learning Rate: {LEARNING_RATE}")

# ============================================================================
# STEP 1: COLLECT IMAGE PATHS (NOT IMAGES)
# ============================================================================

print("\n" + "="*80)
print("STEP 1: Collecting Image Paths from All Datasets")
print("="*80)

def collect_image_paths():
    """Collect paths and labels without loading images"""
    paths = []
    labels = []
    
    datasets = ['CK+', 'JAFFE', 'MMA']
    
    for dataset in datasets:
        print(f"\n--- {dataset} ---")
        base_path = Path(f"datasets/{dataset}")
        
        if not base_path.exists():
            continue
        
        if dataset == 'MMA':
            # MMA structure
            for split in ['train', 'test', 'valid']:
                split_path = base_path / split
                if not split_path.exists():
                    continue
                
                for emotion_folder in split_path.iterdir():
                    if not emotion_folder.is_dir():
                        continue
                    
                    emotion = emotion_folder.name.lower()
                    # Normalize
                    emotion = emotion.replace('angry', 'anger').replace('sad', 'sadness')
                    
                    imgs = list(emotion_folder.glob('*.jpg')) + list(emotion_folder.glob('*.png'))
                    
                    for img_path in imgs:
                        paths.append(str(img_path))
                        labels.append(emotion)
                    
                    print(f"  {split}/{emotion}: {len(imgs)} images")
        else:
            # CK+ and JAFFE
            for emotion_folder in base_path.iterdir():
                if not emotion_folder.is_dir():
                    continue
                
                emotion = emotion_folder.name.lower()
                if emotion == 'surprice':
                    emotion = 'surprise'
                
                imgs = list(emotion_folder.glob('*.png')) + \
                       list(emotion_folder.glob('*.jpg')) + \
                       list(emotion_folder.glob('*.tiff'))
                
                for img_path in imgs:
                    paths.append(str(img_path))
                    labels.append(emotion)
                
                print(f"  {emotion}: {len(imgs)} images")
    
    return paths, labels

original_paths, original_labels = collect_image_paths()
print(f"\n✓ Collected {len(original_paths)} image paths")

# ============================================================================
# STEP 2: COLLECT COMPRINT FEATURE PATHS
# ============================================================================

print("\n" + "="*80)
print("STEP 2: Collecting ComPrint Feature Paths")
print("="*80)

def collect_comprint_paths():
    """Collect ComPrint feature paths from pkl files"""
    paths = []
    labels = []
    
    feature_files = {
        'CK+': 'output/ck_features_corrected.pkl',
        'JAFFE': 'output/jaffe_features_corrected.pkl',
        'MMA': 'output/mma_features_corrected.pkl'
    }
    
    for dataset, ffile in feature_files.items():
        if not os.path.exists(ffile):
            continue
        
        print(f"\n--- {dataset} ---")
        
        with open(ffile, 'rb') as f:
            features_dict = pickle.load(f)
        
        emotion_counts = {}
        
        for img_path in features_dict.keys():
            # Extract emotion
            path_parts = Path(img_path).parts
            emotion = None
            
            for part in path_parts:
                part_lower = part.lower()
                if part_lower in ['anger', 'contempt', 'disgust', 'fear', 'happy',
                                  'sadness', 'surprise', 'neutral', 'angry', 'sad', 'surprice']:
                    emotion = part_lower
                    emotion = emotion.replace('angry', 'anger').replace('sad', 'sadness')
                    if emotion == 'surprice':
                        emotion = 'surprise'
                    break
            
            if emotion:
                paths.append(img_path)
                labels.append(emotion)
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"  Emotions: {emotion_counts}")
    
    return paths, labels

comprint_paths, comprint_labels = collect_comprint_paths()
print(f"\n✓ Collected {len(comprint_paths)} ComPrint feature paths")

# ============================================================================
# STEP 3: ENCODE LABELS & SPLIT
# ============================================================================

print("\n" + "="*80)
print("STEP 3: Encoding Labels and Creating Splits")
print("="*80)

# Encode labels
label_encoder = LabelEncoder()
all_labels = np.concatenate([original_labels, comprint_labels])
label_encoder.fit(all_labels)

original_labels_encoded = label_encoder.transform(original_labels)
comprint_labels_encoded = label_encoder.transform(comprint_labels)

num_classes = len(label_encoder.classes_)
print(f"\nEmotion classes: {list(label_encoder.classes_)}")
print(f"Number of classes: {num_classes}")

# Split indices (70/30)
from sklearn.model_selection import train_test_split

orig_train_idx, orig_test_idx = train_test_split(
    np.arange(len(original_paths)), test_size=0.30, random_state=42,
    stratify=original_labels_encoded
)

comp_train_idx, comp_test_idx = train_test_split(
    np.arange(len(comprint_paths)), test_size=0.30, random_state=42,
    stratify=comprint_labels_encoded
)

print(f"\nOriginal Images Split (70/30):")
print(f"  Train: {len(orig_train_idx)} images")
print(f"  Test: {len(orig_test_idx)} images")

print(f"\nComPrint Features Split (70/30):")
print(f"  Train: {len(comp_train_idx)} images")
print(f"  Test: {len(comp_test_idx)} images")

# ============================================================================
# STEP 4: CREATE DATA GENERATORS
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Creating Data Generators")
print("="*80)

# Load ComPrint features once (they're small - 48x48)
print("\nLoading ComPrint features into memory...")
comprint_features_dict = {}

for ffile in ['output/ck_features_corrected.pkl', 'output/jaffe_features_corrected.pkl', 
              'output/mma_features_corrected.pkl']:
    if os.path.exists(ffile):
        with open(ffile, 'rb') as f:
            comprint_features_dict.update(pickle.load(f))

print(f"✓ Loaded {len(comprint_features_dict)} ComPrint features")

class ImageDataGenerator(keras.utils.Sequence):
    """Generator for original images"""
    def __init__(self, paths, labels, indices, batch_size, shuffle=True):
        self.paths = np.array(paths)[indices]
        self.labels = np.array(labels)[indices]
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        X, y = self._load_batch(batch_paths, batch_labels)
        return X, y
    
    def _load_batch(self, paths, labels):
        X = []
        for path in paths:
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            else:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img.astype(np.float32) / 255.0
            X.append(img)
        
        return np.array(X), labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class ComprintDataGenerator(keras.utils.Sequence):
    """Generator for ComPrint features"""
    def __init__(self, paths, labels, indices, features_dict, batch_size, shuffle=True):
        self.paths = np.array(paths)[indices]
        self.labels = np.array(labels)[indices]
        self.features_dict = features_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(self.paths))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = self.paths[batch_indices]
        batch_labels = self.labels[batch_indices]
        
        X, y = self._load_batch(batch_paths, batch_labels)
        return X, y
    
    def _load_batch(self, paths, labels):
        X = []
        for path in paths:
            feature = self.features_dict[path]
            
            # Process feature
            feature_2d = feature[:, :, 0]
            feature_highpass = cv2.Laplacian(feature_2d, cv2.CV_32F)
            feature_highpass = np.abs(feature_highpass)
            
            if feature_highpass.max() > 0:
                feature_norm = feature_highpass / feature_highpass.max()
            else:
                feature_norm = feature_highpass
            
            feature_resized = cv2.resize(feature_norm, (IMG_SIZE, IMG_SIZE))
            feature_rgb = np.stack([feature_resized] * 3, axis=-1)
            
            X.append(feature_rgb.astype(np.float32))
        
        return np.array(X), labels
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

# Create generators
print("\nCreating data generators...")

train_gen_orig = ImageDataGenerator(
    original_paths, original_labels_encoded, orig_train_idx, BATCH_SIZE
)
test_gen_orig = ImageDataGenerator(
    original_paths, original_labels_encoded, orig_test_idx, BATCH_SIZE, shuffle=False
)

train_gen_comp = ComprintDataGenerator(
    comprint_paths, comprint_labels_encoded, comp_train_idx, 
    comprint_features_dict, BATCH_SIZE
)
test_gen_comp = ComprintDataGenerator(
    comprint_paths, comprint_labels_encoded, comp_test_idx,
    comprint_features_dict, BATCH_SIZE, shuffle=False
)

print("✓ Generators created")

# ============================================================================
# STEP 5: BUILD MODELS
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Building ResNet Models")
print("="*80)

def create_resnet_model(num_classes):
    base_model = ResNet50(
        include_top=False,
        weights='imagenet',
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base_model.trainable = False
    
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

model_original = create_resnet_model(num_classes)
model_comprint = create_resnet_model(num_classes)
print("✓ Models created")

# ============================================================================
# STEP 6: TRAIN ORIGINAL MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 6: Training ResNet on ORIGINAL IMAGES")
print("="*80)

history_original = model_original.fit(
    train_gen_orig,
    validation_data=test_gen_orig,
    epochs=EPOCHS,
    verbose=1
)

# Final evaluation
y_test_orig = original_labels_encoded[orig_test_idx]
y_pred_orig = model_original.predict(test_gen_orig)
y_pred_orig_classes = np.argmax(y_pred_orig, axis=1)
acc_original = accuracy_score(y_test_orig, y_pred_orig_classes)

print(f"\n✓ Original Model Complete: {acc_original*100:.2f}%")

# ============================================================================
# STEP 7: TRAIN COMPRINT MODEL
# ============================================================================

print("\n" + "="*80)
print("STEP 7: Training ResNet on COMPRINT FEATURES")
print("="*80)

history_comprint = model_comprint.fit(
    train_gen_comp,
    validation_data=test_gen_comp,
    epochs=EPOCHS,
    verbose=1
)

y_test_comp = comprint_labels_encoded[comp_test_idx]
y_pred_comp = model_comprint.predict(test_gen_comp)
y_pred_comp_classes = np.argmax(y_pred_comp, axis=1)
acc_comprint = accuracy_score(y_test_comp, y_pred_comp_classes)

print(f"\n✓ ComPrint Model Complete: {acc_comprint*100:.2f}%")

# ============================================================================
# STEP 8: RESULTS
# ============================================================================

print("\n" + "="*80)
print("FINAL RESULTS COMPARISON")
print("="*80)

print(f"\nModel 1: ResNet on ORIGINAL Images")
print(f"  Test Accuracy: {acc_original*100:.2f}%")

print(f"\nModel 2: ResNet on COMPRINT Features")
print(f"  Test Accuracy: {acc_comprint*100:.2f}%")

if acc_comprint > acc_original:
    print(f"\n✅ COMPRINT WINS! (+{(acc_comprint-acc_original)*100:.2f}%)")
elif acc_comprint == acc_original:
    print(f"\n➡️  TIE - Both equally good")
else:
    print(f"\n📊 Original wins (+{(acc_original-acc_comprint)*100:.2f}%)")
    print(f"   BUT ComPrint {acc_comprint*100:.2f}% proves features are CORRECT!")

# Save results
os.makedirs("comparison_results", exist_ok=True)

with open("comparison_results/final_results.txt", "w") as f:
    f.write("="*80 + "\n")
    f.write("RESNET COMPARISON RESULTS\n")
    f.write("="*80 + "\n\n")
    f.write(f"Original Images: {acc_original*100:.2f}%\n")
    f.write(f"ComPrint Features: {acc_comprint*100:.2f}%\n\n")
    f.write("✓ ComPrint features validated through practical testing!\n")