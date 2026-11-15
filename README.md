# TRADD
### Temporal Reliability-Aware Deepfake Detection

TRADD is a deepfake detection framework that combines spatial, frequency-domain, and temporal features to classify videos as REAL or FAKE, with an option to abstain when predictions are uncertain.

## 1. Project Overview

The framework uses three main components:

- **Spatial branch:** MobileNetV3 Small extracts visual features from face frames.
- **Frequency branch:** Discrete Cosine Transform (DCT) captures frequency-domain information.
- **Temporal branch:** GRU models changes across consecutive video frames.

A reliability-based decision mechanism allows the model to classify uncertain predictions as UNCERTAIN.

## 2. Dataset

Experiments were conducted using FaceForensics++ (c23).

- Total videos: 500
- Real videos: 100
- Fake videos: 400
- Frames per video: 10
- Training videos: 300
- Validation videos: 100
- Testing videos: 100

The dataset is not included in this repository.

## 3. Model Performance

| Model | Validation AUC | Test AUC |
|---|---:|---:|
| RGB | 0.8294 | 0.8206 |
| RGB + DCT | 0.8038 | 0.8306 |
| RGB + DCT + GRU | 0.8044 | 0.8225 |

The RGB + DCT model achieved the highest test ROC-AUC of 0.8306.

## 4. Reliability Results

The class-wise reliability mechanism achieved:

- Test coverage: 75%
- Selective accuracy: 90.67%
- Uncertain predictions: 25%

These results are based on validation-selected thresholds. They should not be interpreted as evidence of calibrated uncertainty.

## 5. Technologies Used

- Python
- PyTorch
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- MobileNetV3
- GRU
- DCT

## 6. Project Structure

```text
TRADD/
├── datasets/
├── models/
├── final_models/
├── notebooks/
├── src/
│   ├── models/
│   ├── training/
│   └── evaluation/
├── results/
│   ├── figures/
│   └── tables/
├── reports/
├── docs/
├── README.md
├── requirements.txt
└── .gitignore
