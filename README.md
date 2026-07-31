# Crop Disease Detection using CNNs

A deep learning project focused on detecting and classifying crop diseases from leaf images using Convolutional Neural Networks (CNNs). The project utilizes the extensive PlantVillage dataset to categorize leaves into 38 distinct classes of healthy and diseased crops.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Setup & Installation](#setup--installation)
- [Project Workflow](#project-workflow)

## Overview
Early detection of crop diseases is crucial for ensuring food security and maximizing agricultural yield. This notebook provides a complete pipeline to automatically identify crop diseases from color images of leaves. The pipeline includes data acquisition, visualization, and deep learning model setup using PyTorch.

## Dependencies
The project requires the following libraries. It is highly recommended to run this project in an environment with GPU support (such as Google Colab).
- `torch`
- `torchvision`
- `matplotlib`
- `scikit-learn`
- `seaborn`
- `numpy`
- `Pillow` (PIL)
- `kaggle`

To install the dependencies, run:
```bash
pip install torch torchvision matplotlib scikit-learn seaborn numpy pillow kaggle --quiet
```

## Dataset
This project uses the **PlantVillage Dataset** hosted on Kaggle (`abdallahalidev/plantvillage-dataset`). 
It features a comprehensive collection of leaf images spanning **38 disease classes** (e.g., Apple Scab, Apple Black rot, Cedar apple rust, Blueberry healthy, etc.).

## Setup & Installation

### 1. Kaggle API Configuration
To download the dataset directly, you must configure your Kaggle API credentials.

1. Go to your Kaggle Profile -> Settings -> API -> Create New Token to download `kaggle.json`.
2. Place the `kaggle.json` file in the `~/.kaggle/` directory and set appropriate permissions:
   ```bash
   mkdir -p ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

### 2. Download the Dataset
Once configured, you can pull the dataset directly via the Kaggle CLI:
```bash
kaggle datasets download -d abdallahalidev/plantvillage-dataset
unzip plantvillage-dataset.zip -d plantvillage
```
The images will be extracted to `plantvillage/plantvillage dataset/color`.

## Project Workflow
The current notebook is structured into the following initial stages:

1. **Environment Setup & Imports**: Imports required Python libraries and detects available hardware accelerators (CUDA/GPU).
2. **Data Acquisition**: Automates the configuration of the Kaggle API, downloads the 2.04GB PlantVillage dataset, and extracts it to reveal 38 distinct disease classes.
3. **Data Visualization**: Iterates through the directory structure to extract sample images across different classes and visualizes them using a `matplotlib` 3x4 grid for preliminary analysis.