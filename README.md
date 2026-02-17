# 🎭 ComPrint Features for Facial Emotion Recognition

## 📋 Project Overview

This research project investigates whether **ComPrint** — a compression fingerprint extraction model originally designed for image forgery detection — can serve as a meaningful feature space for **facial emotion recognition**.

ComPrint uses a Siamese neural network to detect JPEG compression artifacts. We hypothesize that these compression patterns may encode subtle structural and texture cues related to facial expressions, providing an alternative feature representation for emotion classification.

### Key Question
*Can compression fingerprints alone — without direct facial features — enable emotion recognition better than random chance?*

---

## 🎯 Results Summary

| Model | Input Type | Test Accuracy | vs. Random Baseline |
|-------|-----------|---------------|---------------------|
| **ResNet50** | Original Images | **37.46%** | 3.4× better |
| **ResNet50** | ComPrint Features | **33.41%** | 3.0× better |
| Random Guess | — | 11.11% | — |

### Key Findings

✅ **ComPrint features significantly outperform random chance** (33.41% vs 11.11%), confirming they encode genuine emotion-relevant information

✅ **4% accuracy gap** between original images and ComPrint features is expected — ComPrint operates on 48×48 compression residuals while original images contain full facial detail

✅ **Validation successful** — all 128,874 features extracted with 100% success rate, re-extraction test showed perfect consistency (0.000000 difference)

---

## 📊 Datasets

| Dataset | Images | Classes | Source |
|---------|--------|---------|--------|
| **CK+** | 981 | 7 emotions | [Extended Cohn-Kanade](http://www.jeffcohn.net/Resources/) |
| **JAFFE** | 213 | 7 emotions | [Japanese Female Facial Expression](https://zenodo.org/record/3451524) |
| **MMA** | 127,680 | 7 emotions | Facial Expression Dataset |
| **Total** | **128,874** | 7 classes | — |

**Emotion Classes:** Anger, Contempt, Disgust, Fear, Happy, Sadness, Surprise

---

## 🏗️ Project Structure

```
emotion-recognition-comprint/
├── 📄 README.md                          # This file
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                         # Git ignore rules
│
├── 📁 datasets/                          # Raw image datasets (not in repo)
│   ├── CK+/
│   ├── JAFFE/
│   └── MMA/
│
├── 📁 comprint/                          # ComPrint model (not in repo)
│   └── models/
│
├── 🐍 fixed_extract_final.py             # Extract CK+ & JAFFE features
├── 🐍 extract_mma_features.py            # Extract MMA features
├── 🐍 organize_by_class.py               # Organize features by emotion
├── 🐍 cnn_train_orignal_image.py         # Train CNN on original images
├── 🐍 resnet_comparison.py               # Compare Original vs ComPrint
├── 🐍 generate_complete_results.py       # Generate result visualizations
│
├── 📓 CNN_With_Comprint.ipynb            # Jupyter: ComPrint training
├── 📓 CNN_With_orignal_images.ipynb      # Jupyter: Original image training
│
└── 📁 output/                            # Generated results (not in repo)
    ├── ck_features_corrected.pkl
    ├── jaffe_features_corrected.pkl
    ├── mma_features_corrected.pkl
    └── visualizations_by_class/
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- TensorFlow 2.9.1
- 16GB+ RAM recommended (for MMA dataset processing)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/YOUR_USERNAME/comprint-emotion-recognition.git
cd comprint-emotion-recognition
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download datasets:**
- [CK+ Dataset](http://www.jeffcohn.net/Resources/)
- [JAFFE Dataset](https://zenodo.org/record/3451524)
- Place in `datasets/CK+/` and `datasets/JAFFE/`

4. **Download ComPrint model:**
```bash
git clone https://github.com/IDLabMedia/comprint.git
```

---

## 📖 Usage

### Step 1: Extract ComPrint Features
```bash
# Extract CK+ and JAFFE features
python fixed_extract_final.py

# Extract MMA features (takes longer - 127K images)
python extract_mma_features.py
```

### Step 2: Organize Features by Emotion
```bash
python organize_by_class.py
```

### Step 3: Train Models
```bash
# Train CNN on original images
python cnn_train_orignal_image.py

# Run ResNet comparison (Original vs ComPrint)
python resnet_comparison.py
```

### Step 4: Generate Result Visualizations
```bash
python generate_complete_results.py
```

---

## 📈 Model Architecture

### ResNet50 Transfer Learning
- **Base Model:** ResNet50 pre-trained on ImageNet
- **Input Size:** 224×224×3
- **Frozen Layers:** All ResNet50 layers
- **Custom Head:** 
  - GlobalAveragePooling2D
  - Dense(256, activation='relu')
  - Dropout(0.5)
  - Dense(num_classes, activation='softmax')
- **Optimizer:** Adam (learning_rate=0.0001)
- **Training:** 15 epochs, batch_size=32

---

## 📊 Results Breakdown

### CNN Training on Original Images (CK+ Dataset)

| Emotion | Precision | Recall | F1-Score | Support |
|---------|-----------|--------|----------|---------|
| **Surprise** | 0.59 | 0.76 | 0.66 | 49 |
| **Contempt** | 0.47 | 0.70 | 0.56 | 10 |
| **Happy** | 0.27 | 0.49 | 0.35 | 41 |
| Anger | 0.26 | 0.26 | 0.26 | 27 |
| Disgust | 0.40 | 0.17 | 0.24 | 35 |
| Fear | 0.00 | 0.00 | 0.00 | 15 |
| Sadness | 0.00 | 0.00 | 0.00 | 16 |
| **Overall** | **0.34** | **0.40** | **0.35** | **193** |

**Observation:** Surprise and Contempt performed best, likely due to distinctive facial patterns. Fear and Sadness struggled, possibly due to subtle expressions and class imbalance.

---

## 🔬 Validation Methodology

To verify ComPrint features are correctly extracted and meaningful:

1. **Re-extraction Test:** Re-extracted random samples → 0.000000 difference (perfect match)
2. **Visual Inspection:** Compression heatmaps show visible JPEG artifacts
3. **Statistical Validation:** Features significantly outperform random baseline (3× better)
4. **Classification Test:** ResNet successfully learns from features (33.41% accuracy)

**Conclusion:** ComPrint features are correctly extracted and encode genuine patterns relevant to emotion recognition, though not surpassing original images (expected due to information loss).

---

## 📚 Citation

### This Project
If you use this work, please cite:
```bibtex
@misc{comprint_emotion_recognition,
  author = {Your Name},
  title = {ComPrint Features for Facial Emotion Recognition},
  year = {2024},
  url = {https://github.com/YOUR_USERNAME/comprint-emotion-recognition}
}
```

### ComPrint Model
This project uses the ComPrint model. Please cite the original authors:
```bibtex
@InProceedings{mareen2022comprint,
  author = {Mareen, Hannes and Vanden Bussche, Dante and Guillaro, Fabrizio and 
            Cozzolino, Davide and Van Wallendael, Glenn and Lambert, Peter and Verdoliva, Luisa},
  title = {Comprint: Image Forgery Detection and Localization Using Compression Fingerprints},
  booktitle = {Pattern Recognition, Computer Vision, and Image Processing. ICPR 2022},
  year = {2023},
  publisher = {Springer Nature Switzerland},
  pages = {281--299},
  doi = {10.1007/978-3-031-37742-6_23}
}
```

---

## 📄 License

- **This Project:** MIT License (see LICENSE)
- **ComPrint Model:** Research-only license by IDLab-MEDIA and GRIP-UNINA
  - ✅ Allowed: Academic research, non-commercial use
  - ❌ Prohibited: Commercial applications without permission
  - See ComPrint's [LICENSE.txt](https://github.com/IDLabMedia/comprint/blob/main/LICENSE.txt)

---

## 🤝 Acknowledgments

- **ComPrint Model:** [IDLab-MEDIA, Ghent University](https://media.idlab.ugent.be/) and [GRIP-UNINA](https://www.grip.unina.it/)
- **Datasets:** CK+, JAFFE, and MMA dataset contributors
- **Framework:** TensorFlow, Keras, scikit-learn

---

## 📧 Contact

- **Author:** Muhamad Yousha
- **Email:** youshalashari@gmail.com
- **LinkedIn:** [[Your LinkedIn Profile](https://linkedin.com/in/yourprofile)](https://www.linkedin.com/in/muhammad-yousha-484b2a362/)

---

## 🔗 Useful Links

- [ComPrint Paper (arXiv)](https://arxiv.org/abs/2210.02227)
- [ComPrint GitHub](https://github.com/IDLabMedia/comprint)
- [CK+ Dataset](http://www.jeffcohn.net/Resources/)
- [JAFFE Dataset](https://zenodo.org/record/3451524)
