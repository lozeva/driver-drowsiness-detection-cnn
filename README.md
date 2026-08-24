# Driver Drowsiness Detection: A CNN and Transfer Learning Comparison

## Overview
This project investigates whether a driver's facial image can be classified as **Drowsy** or **Non-Drowsy** using deep learning. Two approaches are compared - a custom CNN trained from scratch and a MobileNetV2 transfer learning model - using a **subject-level** train/validation/test split (not image-level) to avoid data leakage between individuals.

## Dataset
**Driver Drowsiness Dataset (DDD)** - [Kaggle](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd/code)
- 41,793 facial images, 54 unique subjects
- Labels assigned at the subject/recording-session level: each subject belongs entirely to one class (28 drowsy-only, 26 non-drowsy-only)

The raw dataset is not included in this repo (see `.gitignore`) - download it from Kaggle and place it under `data/raw/Driver Drowsiness Dataset (DDD)/` before running the notebook.

## Project Structure
```
driver-drowsiness-detection-cnn/
├── data/raw/                          # DDD dataset (not tracked in git)
├── src/
│   ├── data_utils.py                  # file indexing, subject-level split, data generators
│   ├── models.py                      # build_custom_cnn(), build_mobilenetv2_model()
│   └── evaluation.py                  # training curves, confusion matrices, classification reports
├── models/                            # saved .keras model weights
├── reports/figures/                   # saved plots
├── driver_drowsiness_detection.ipynb  # main notebook
├── requirements.txt
└── README.md
```
> Preprocessing (resize, normalize, augment) is done on-the-fly via Keras `ImageDataGenerator`, not persisted to disk.

## Models
- **Custom CNN**: 3 conv blocks (Conv2D -> BatchNorm -> ReLU -> MaxPooling, 32->64->128), GlobalAveragePooling2D, Dense(128)+Dropout head. 109,761 trainable / 448 non-trainable params.
- **MobileNetV2**: frozen ImageNet base, GlobalAveragePooling2D, Dense(128)+Dropout head. 164,103 trainable / 2,257,984 non-trainable params.

Both: Adam optimizer, binary cross-entropy loss, `EarlyStopping`(patience=3) + `ModelCheckpoint`. Trained on CPU (`epochs=10`).

## Results

Evaluated on 12 unseen test subjects (6,338 images):

| Model | Test Accuracy | Precision (Drowsy) | Recall (Drowsy) | F1 (Drowsy) |
|---|:---:|:---:|:---:|:---:|
| Custom CNN | 35% | 0.19 | 0.07 | 0.10 |
| MobileNetV2 | 34% | 0.36 | 0.31 | 0.33 |

Both models reached ~97-99% **training** accuracy but fell below the 52.8% majority-class baseline on test data. A second training run (same setup, different random seed) previously gave 60%/53% test accuracy - performance is highly sensitive to random initialization given only 12 test subjects.

## Key Finding

Since `subject_id` perfectly determines `drowsiness_label` in DDD, both models likely overfit to subject-specific visual characteristics (background, lighting, facial identity) rather than learning generalizable drowsiness cues. This was investigated methodically: identified during data exploration and quality checks (Section 5), and confirmed in training curves and test evaluation (Sections 7-8). A preprocessing-mismatch hypothesis for MobileNetV2 was also tested; correcting it did not improve the results, suggesting that this preprocessing issue was not the primary explanation for the observed generalization problem, though the exact cause cannot be established with certainty from the available experiments. Results are also substantially lower than related work (96-99.97% reported accuracy). The studies use different datasets and methodologies, and their splitting strategies may differ - higher reported accuracy should therefore not be directly compared without verifying their evaluation protocols. See the notebook (Sections 9-10) for full discussion.

## Limitations
- Only 54 subjects total (31 train / 11 val / 12 test) - little diversity for generalization
- CPU-only training constrained the epoch budget and experimentation
- No fixed random seed - substantial run-to-run variability in exact numbers
- Session-level (not frame-level) labeling introduces some label noise
- Single split rather than subject-level k-fold cross-validation

## Future Work
A larger dataset with more subjects per class (or subjects contributing both classes), MobileNetV2 fine-tuning instead of a fully frozen base, stronger regularization/augmentation, fixed random seeds, and subject-level k-fold cross-validation.

## Technologies
Python, Pandas, NumPy, TensorFlow/Keras, Scikit-learn, Matplotlib, Seaborn, Jupyter

## How to Run
```bash
git clone https://github.com/lozeva/driver-drowsiness-detection-cnn.git
cd driver-drowsiness-detection-cnn
pip install -r requirements.txt
```
Download the dataset from Kaggle into `data/raw/Driver Drowsiness Dataset (DDD)/`, then:
```bash
jupyter notebook
```
Open `driver_drowsiness_detection.ipynb`.

> Training runs on CPU only and no random seed is fixed, so re-running may produce different exact numbers, though the overfitting pattern is expected to persist.