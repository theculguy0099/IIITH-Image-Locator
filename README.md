<p align="center">
    <img src="https://raw.githubusercontent.com/PKief/vscode-material-icon-theme/ec559a9f6bfd399b82bb44393651661b08aaf7ba/icons/folder-markdown-open.svg" align="center" width="30%">
</p>
<p align="center"><h1 align="center">IIITH-IMAGE-LOCATOR</h1></p>

<p align="center"><!-- default option, no dependency badges. -->
</p>
<p align="center">
	<!-- default option, no dependency badges. -->
</p>
<br>

##  Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
  - [Project Index](#project-index)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Testing](#testing)
- [Approach](#approach)
- [Results & Evaluation](#results--evaluation)
- [Project Roadmap](#project-roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

##  Overview

IIITH-Image-Locator is a computer vision project designed to predict the precise geographic location (latitude and longitude), camera orientation (angle), and campus region ID (1–15) of images captured within the IIIT Hyderabad campus. The system leverages a manually annotated dataset consisting of 55 images per region, collected across 15 color-coded campus regions. Each image is accompanied by metadata: GPS coordinates, capture timestamp, and orientation angle.

Phase 1 covered data collection and annotation. Phase 2 focuses on model development and evaluation, culminating in a Kaggle contest for final submissions.

---

##  Features

- **Multi-task Prediction**: Simultaneously infer latitude, longitude, camera orientation angle, and region ID from a single image.
- **Data Annotation Tools**: Scripts to standardize images to 256×256 resolution and aggregate metadata into CSV format.
- **Modular Pipelines**: Separate notebooks and models for each target (latitude-longitude regression, angle regression, and region classification).
- **Flexible Model Integration**: Swap between CNN-based backbones or transformer architectures easily.
- **Local Validation**: Pre-built validation set and evaluation scripts for MSE (regression) and accuracy (classification).

---

##  Project Structure

```text
Campus-Image-Locator/
├── direction/            # Angle estimation notebooks
│   ├── angle1.ipynb      # Baseline CNN approach for camera orientation
│   ├── angle2.ipynb      # Transformer-based angle regression
│   └── solution.csv      # Inference outputs for test set
├── latlong/              # Latitude & longitude regression notebooks
│   ├── latlong1.ipynb    # CNN regression baseline
│   ├── latlong2.ipynb    # Data augmentation experiments
│   ├── latlong3.ipynb    # Ensemble of multiple models
│   ├── test-15-models.ipynb # Comparative evaluation of candidate models
│   └── solution.csv      # Inference outputs for test set
├── regionID/             # Region classification notebooks
│   ├── region1.ipynb     # ResNet classifier for region ID
│   └── solution.csv      # Inference outputs for test set
├── models.txt            # Summary of model architectures and hyperparameters
├── scripts/              # Utility scripts for preprocessing and evaluation
│   ├── resize_images.py  # Resize all images to 256×256
│   └── evaluate.py       # Compute MSE and accuracy on validation set
└── README.md             # Project overview and instructions
````

### Project Index

<details open>
<summary><b>IIITH-IMAGE-LOCATOR/</b></summary>

<details>
<summary><b>__root__</b></summary>

| File         | Description                                          |
| ------------ | ---------------------------------------------------- |
| `models.txt` | List of all networks tested with key hyperparameters |

</details>

<details>
<summary><b>latlong/</b></summary>

| Notebook               | Description                                         |
| ---------------------- | --------------------------------------------------- |
| `test-15-models.ipynb` | Comparison of 15 candidate regression architectures |
| `latlong1.ipynb`       | Baseline CNN regression model                       |
| `latlong2.ipynb`       | Experiments with data augmentation                  |
| `latlong3.ipynb`       | Model ensembling and blending strategies            |

</details>

<details>
<summary><b>regionID/</b></summary>

| Notebook        | Description                                 |
| --------------- | ------------------------------------------- |
| `region1.ipynb` | ResNet classifier for campus region mapping |

</details>

<details>
<summary><b>direction/</b></summary>

| Notebook       | Description                             |
| -------------- | --------------------------------------- |
| `angle1.ipynb` | CNN-based angle regression baseline     |
| `angle2.ipynb` | Transformer-based orientation estimator |

</details>

</details>

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* Jupyter Notebook / JupyterLab
* CUDA-enabled GPU (optional for local training)
* Dependencies listed in `requirements.txt`

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/theculguy0099/IIITH-Image-Locator.git
   cd IIITH-Image-Locator
   ```

2. **Create environment and install dependencies**:

   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   pip install -r requirements.txt
   ```

### Usage

* **Preprocess images**:

  ```bash
  python scripts/resize_images.py --input_folder "SMAI Images" --output_folder "256x256"
  ```

* **Run angle estimation**:

  Launch `direction/angle1.ipynb` or `direction/angle2.ipynb` in Jupyter.

* **Run latitude-longitude regression**:

  Open the notebooks under `latlong/` for training and validation.

* **Run region classification**:

  Execute `regionID/region1.ipynb` in Jupyter.

### Testing

After training or loading saved weights, run:

```bash
python scripts/evaluate.py --val_csv path/to/validation.csv --pred_dir path/to/solution-files
```

This will compute MSE for latitude, longitude, angle, and accuracy for region ID.

---


## Approach

This project employs a modular, multi-stage pipeline leveraging state-of-the-art deep learning techniques and recent self-supervised innovations:

1. **Data Preprocessing & Advanced Augmentation**:

   * Standardization of image resolution (256×256) and normalization.
   * Geometric transformations (random rotations, flips, cropping, perspective warps) for viewpoint robustness.
   * Photometric augmentations (random brightness, contrast, hue, and Gaussian noise) to simulate diverse lighting and weather conditions.
   * MixUp and CutMix strategies to improve generalization and reduce label noise.

2. **Self-Supervised Feature Extraction with DINOv2**:

   * Leverage Facebook Research’s DINOv2 (self-distillation with no labels) to extract rich, context-aware embeddings.
   * Freeze or fine-tune DINOv2 backbone for downstream tasks, reducing the need for large labeled datasets.

3. **Latitude & Longitude Regression**:

   * Attach MLP heads on top of DINOv2 embeddings and EfficientNet backbones for coordinate regression.
   * Use Mean Squared Error (MSE) loss optimized with AdamW and cosine annealing learning rate schedules.
   * Ensemble multiple regressor heads (e.g., XGBoost on frozen embeddings + neural MLP) to reduce variance.

4. **Angle Estimation**:

   * Predict camera orientation angle using specialized regression heads on DINOv2 features.
   * Angular loss combining MSE with a circular distance metric to handle wrap-around (0°–360°).
   * Ensemble predictions from CNN-based and transformer-based regressors to smooth out outliers.

5. **Region Classification**:

   * Fine-tune DINOv2 with a classification head for 15-way region ID prediction.
   * Cross-entropy loss with label smoothing, augmented with focal loss to address challenging overlap regions.
   * Ensemble classifiers (e.g., SGD-trained ResNet, DenseNet, and DINOv2-based head) via weighted majority voting.

6. **Multi-task Learning & Joint Optimization**:

   * Unified architecture sharing DINOv2 encoder and separate heads for coordinates, angle, and region.
   * Weighted multi-task loss balancing regression and classification objectives, tuned via hyperparameter search.

7. **Validation & Evaluation**:

   * Local cross-validation with stratified splits to ensure spatial diversity.
   * Early stopping and model checkpointing based on combined validation score.
   * Final model ensemble selection guided by performance on the Kaggle leaderboard.

---

## Results & Evaluation

* **Latitude & Longitude MSE:** 13500 (Scaled values of Latitude and Longitude in range of 100000)
* **Angle Mean Angular Error:** 17.5 degrees
* **Region ID Accuracy:** *97.56%*

Note: A Bonus of 20%(of Project weightage) was awarded for getting the best 20 results in all three tasks, on the Kaggle leaderboard. Also I was the person getting the overall score combining all three tasks in a class of 130.


> **Note:** Got a total of 11.91/20 in the project

---

### Qualitative Analysis
The Campus was divided into 15 distinct regions, each with unique visual characteristics. The model successfully learned to differentiate these regions based on color, architecture, and landmarks. It achieved high accuracy in region classification, with some misclassifications occurring at region boundaries where visual features overlap.

The Laitude and Longitude regression models demonstrated reasonable accuracy, with some outliers in areas with similar visual features. The angle regression model showed good performance, but occasional mispredictions were noted in images with complex orientations.


---

## Project Roadmap

* [x] Data collection & annotation pipeline
* [x] Improve coordinate regression through advanced augmentation
* [x] Experiment with multi-task learning model combining all outputs
* [ ] Deploy best model on a simple web interface
* [ ] Publish final write-up and open-source dataset split scripts

---

## Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Open a pull request
---

## Acknowledgments

* IIIT Hyderabad for dataset framework
* Kaggle for hosting the challenge and leaderboard. 
* OpenCV and PyTorch communities for tooling and support

---
