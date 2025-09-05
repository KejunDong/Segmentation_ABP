# U-Net Training Pipeline for 4-Class Signal Segmentation (PyTorch)

This repository contains a complete pipeline for training a U-Net model to perform 4-class segmentation on arterial blood pressure (ABP) signals using PyTorch.

## Overview

The pipeline is designed to:
- Perform 4-class segmentation on ABP signals
- Use 10-fold cross-validation for robust model evaluation
- Handle severe class imbalance with weighted loss functions
- Provide comprehensive evaluation metrics for each class

## Data Structure

- **Input**: `sample_data.npy` with shape (2, 6800, 400)
  - First dimension: Input signals (6800 segments × 400 samples @ 200 Hz)
  - Second dimension: Label data (6800 segments × 400 samples)
- **4-Class Labels** (after preprocessing):
  - 0: Nothing (background) - 86.20% of data
  - 1: Diastolic chest compression (target class) - 2.87% of data
  - 2: Systolic events (chest compression + normal heartbeat) - 6.65% of data
  - 3: Diastolic normal heartbeat - 4.28% of data

**Note**: Original labels 2 and 4 are combined into class 2 during preprocessing.

## Files Description

### Core Scripts

1. **`main_pytorch.py`** - Main execution script (PyTorch version)
   - Command-line interface for PyTorch pipeline
   - Complete pipeline orchestration
   - 10-fold cross-validation training

2. **`unet_model_pytorch.py`** - U-Net architecture implementation (PyTorch)
   - 1D U-Net adapted for signal segmentation
   - Weighted CrossEntropyLoss for class imbalance
   - Custom metrics (Dice coefficient, IoU)
   - Training and evaluation methods

3. **`data_preprocessing.py`** - Data preprocessing and cross-validation setup
   - 4-class label mapping (combines labels 2 and 4)
   - Signal normalization
   - Creates 10-fold cross-validation splits
   - Saves preprocessed data

4. **`training_pipeline_simple.py`** - Training pipeline (PyTorch)
   - 10-fold cross-validation training
   - Model evaluation and metrics calculation
   - Results saving and statistics

5. **`generate_overall_results.py`** - Results aggregation
   - Combines results from all folds
   - Calculates overall statistics
   - Generates summary reports

### Utility Scripts

6. **`analyze_class_distribution.py`** - Detailed class analysis
   - Analyzes class distribution and imbalance
   - Identifies potential data issues

7. **`requirements_pytorch.txt`** - Python dependencies (PyTorch)

## Usage

### Prerequisites

Install required packages:
```bash
pip install -r requirements_pytorch.txt
```

### Quick Start

1. **Run Complete Pipeline**:
   ```bash
   python3 main_pytorch.py --epochs 100 --batch_size 16
   ```

2. **Run with Custom Parameters**:
   ```bash
   python3 main_pytorch.py --epochs 50 --batch_size 32 --device cuda
   ```

3. **Analyze Data Distribution**:
   ```bash
   python3 analyze_class_distribution.py
   ```

4. **Generate Results Summary**:
   ```bash
   python3 generate_overall_results.py
   ```

### Command Line Options

- `--data_path`: Path to sample_data.npy (default: sample_data.npy)
- `--output_dir`: Output directory for results (default: results_pytorch)
- `--use_all_segments`: Use all segments (default: True)
- `--epochs`: Number of training epochs (default: 100)
- `--batch_size`: Batch size for training (default: 16)
- `--use_augmentation`: Enable data augmentation (currently disabled)
- `--augmentation_factor`: Data augmentation factor (default: 2)
- `--balance_dataset`: Balance dataset (currently disabled)
- `--device`: Device to use (cpu, cuda, or auto)

## Pipeline Steps

1. **Data Preprocessing**:
   - Load and preprocess sample_data.npy
   - Map 5-class labels to 4-class labels
   - Normalize signals (zero mean, unit variance)
   - Create 10-fold cross-validation splits

2. **Model Training**:
   - Train U-Net on each fold using PyTorch
   - Use weighted CrossEntropyLoss for class imbalance
   - Apply early stopping and learning rate reduction
   - Save best models for each fold

3. **Evaluation**:
   - Calculate comprehensive metrics (accuracy, precision, recall, F1, Dice, IoU)
   - Generate per-class and macro-averaged results
   - Save detailed results and model checkpoints

## Results

The pipeline generates:
- **Models**: Saved PyTorch U-Net models for each fold
- **Metrics**: Detailed evaluation metrics (JSON and TXT format)
- **Summary**: Overall cross-validation statistics
- **Analysis**: Class distribution and performance analysis

### Sample Results

**Overall Performance (10-fold CV)**:
- Accuracy: 97.58% ± 0.27%
- Macro F1: 92.21% ± 0.74%
- Macro Dice: 92.21% ± 0.74%
- Macro IoU: 86.13% ± 1.16%

**Per-Class Performance**:
- Class 0 (nothing): F1 = 98.60% ± 0.16%
- Class 1 (diastolic_CC): F1 = 83.57% ± 1.47% ← Target class
- Class 2 (systolic): F1 = 96.95% ± 0.23%
- Class 3 (diastolic_NH): F1 = 89.72% ± 1.28%

## Key Features

- **10-fold Cross-Validation**: Robust evaluation with proper segment separation
- **Class Imbalance Handling**: Weighted CrossEntropyLoss with class weights
- **Comprehensive Metrics**: Per-class and macro-averaged evaluation metrics
- **PyTorch Implementation**: Modern deep learning framework
- **Modular Design**: Easy to modify and extend

## Model Architecture

The 1D U-Net architecture includes:
- **Encoder**: 4 convolutional blocks with max pooling
- **Bottleneck**: Deep feature extraction layer
- **Decoder**: 4 upsampling blocks with skip connections
- **Output**: 4-class softmax activation
- **Regularization**: Batch normalization and ReLU activations

## Evaluation Metrics

- **Overall**: Accuracy, Macro Precision, Macro Recall, Macro F1
- **Segmentation**: Dice Coefficient, Intersection over Union (IoU)
- **Per-Class**: Precision, Recall, F1, Dice, IoU for each class
- **Averaging**: Macro averaging (equal weight per class)

## Class Imbalance Analysis

The dataset exhibits severe class imbalance:
- **Class 0 (nothing)**: 86.20% of data points
- **Class 1 (diastolic_CC)**: 2.87% of data points (target class)
- **Class 2 (systolic)**: 6.65% of data points
- **Class 3 (diastolic_NH)**: 4.28% of data points

**Handling Strategy**:
- Weighted CrossEntropyLoss with class weights [1.0, 10.0, 3.0, 5.0]
- Focus on per-class metrics rather than overall accuracy
- Macro averaging for balanced evaluation across classes

## Performance Analysis

The model achieves good performance despite severe class imbalance:
- High accuracy (97.58%) primarily due to majority class dominance
- Good per-class performance, especially for the target class (F1=83.57%)
- Effective handling of rare classes through weighted loss


