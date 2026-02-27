Brain Tumor MRI Detection & Classification (PyTorch)
=====================================================

Deep learning project for brain tumor detection (binary classification)
and tumor type classification (multi-class) using a custom VGG16 architecture.

The project performs:
1. Tumor Detection (Tumor vs No Tumor)
2. Tumor Classification (Glioma, Meningioma, Pituitary)

All models are trained from scratch using PyTorch.

---------------------------------------------------------------------

Overview
--------

This project implements a full deep learning pipeline including:

- Dataset download from Kaggle
- Custom image augmentation
- Train / validation / test splitting
- Custom Dataset and DataLoader classes
- Custom VGG16 CNN implementation
- Binary classification (tumor detection)
- Multi-class classification (tumor type prediction)
- Early stopping
- Performance visualization
- Confusion matrix and classification report evaluation

The model operates on grayscale MRI images resized to 240x240.

---------------------------------------------------------------------

Datasets
--------

Detection Dataset:
- Kaggle: mohamada2274/brain-tumor-mri

Classification Dataset:
- Kaggle: masoudnickparvar/brain-tumor-mri-dataset

Datasets are downloaded programmatically using kagglehub.

Note:
Some folder naming inconsistencies (e.g., "Brian" instead of "Brain")
are handled manually in the pipeline.

---------------------------------------------------------------------

Data Augmentation
-----------------

Custom augmentation pipeline includes:

- Random rotation (-20° to +20°)
- Horizontal flip (50% probability)
- Brightness adjustment
- Contrast adjustment
- Color enhancement
- Random scaling (0.8x to 1.2x)

Augmented images are saved locally before training.

This is done separately for:
- Detection model
- Classification model

---------------------------------------------------------------------

Model Architecture
------------------

Custom VGG16-style Convolutional Neural Network:

Feature Extractor:
- 5 convolutional blocks
- Conv2D + ReLU layers
- MaxPooling layers
- AdaptiveAvgPool to (7x7)

Classifier:
- Fully connected layers (4096 → 4096 → output)
- Dropout
- Kaiming weight initialization

Detection model:
- Output classes: 2 (Tumor / No Tumor)

Classification model:
- Output classes: 4 (No Tumor, Glioma, Meningioma, Pituitary)

---------------------------------------------------------------------

Training Configuration
----------------------

Optimizer: Adam  
Learning Rate: 1e-4  
Loss Function: CrossEntropyLoss  
Batch Size: 64  
Epochs: 40  
Early Stopping Patience: 5  

GPU acceleration is used if available.

---------------------------------------------------------------------

Evaluation Metrics
------------------

- Accuracy
- Confusion Matrix
- Classification Report (Precision, Recall, F1-Score)
- Training vs Validation Loss curves
- Training vs Validation Accuracy curves

For detection:
Labels are clamped to binary (0 = No Tumor, 1 = Tumor).

For classification:
Full multi-class labels are used.

---------------------------------------------------------------------

Project Structure (Conceptual)
------------------------------

- Data Augmentation Functions
- Custom Dataset class
- Data Splitting (train/val/test)
- VGG16 Model Definition
- Training Loop with Early Stopping
- Evaluation + Metrics
- Visualization of Results

---------------------------------------------------------------------

Technical Stack
---------------

- Python
- PyTorch
- Torchvision
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- PIL
- KaggleHub

---------------------------------------------------------------------

Key Concepts Demonstrated
-------------------------

- CNN architecture design
- Manual VGG16 implementation
- Weight initialization (Kaiming)
- Custom dataset handling
- Data augmentation pipelines
- Binary vs multi-class classification
- Early stopping implementation
- GPU training workflows
- Performance analysis using confusion matrices

---------------------------------------------------------------------

Limitations
-----------

- Training from scratch (no transfer learning)
- No model checkpoint saving
- Augmented images are pre-saved instead of using online transforms
- No hyperparameter search
- No cross-validation

---------------------------------------------------------------------

Future Improvements
-------------------

- Use pretrained VGG16 (transfer learning)
- Implement model checkpoint saving
- Add learning rate scheduler
- Replace manual augmentation with torchvision transforms
- Add Grad-CAM visualization for interpretability
- Perform hyperparameter tuning
- Deploy model via Flask / FastAPI

---------------------------------------------------------------------

Purpose
-------

This project demonstrates end-to-end deep learning model development
for medical image classification, including preprocessing,
architecture implementation, training, evaluation, and analysis.

It highlights strong understanding of:

- Convolutional neural networks
- Data pipelines
- Model training lifecycle
- Performance evaluation
- PyTorch-based deep learning workflows
