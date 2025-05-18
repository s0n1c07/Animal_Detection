# Animal Detection with PyTorch

A PyTorch-based Convolutional Neural Network (CNN) for classifying animal images. This notebook trains a custom CNN with label smoothing and data augmentation, then generates predictions on a held-out test set, saving them in `submission.csv`.

## Project Overview

This project demonstrates how to build and train a CNN from scratch in PyTorch to classify animal images. It includes:

- Custom data augmentations (cropping, flips, rotations, color jitter).  
- A bespoke CNN architecture with batch normalization and SiLU activations.  
- Label smoothing for more robust training.  
- A cyclic learning‑rate scheduler to help with convergence.  

All steps—from data loading through training to test‑set prediction—are contained in `animal-detection.ipynb`.

---

## Dataset

The dataset was used from kaggle.

## Requirements
Python ≥ 3.7

PyTorch ≥ 1.8

torchvision

pandas

numpy

Pillow

## Model Architecture

Custom CNN

Convolutional blocks with Conv2d → BatchNorm2d → SiLU → Conv2d → BatchNorm2d → SiLU → MaxPool.

Final fully connected layers producing logits for each class.

Loss

Label‑smoothing cross‑entropy (smoothing=0.1).

## Training Details

Hyperparameters

Epochs: 10

Batch size: 16

Learning rate: 1e‑3

Image size: 380×380

Optimizer

AdamW with weight decay 1e‑5.

Scheduler

CyclicLR (triangular2) between 1e‑5 and 1e‑3, step up over 5 iterations.

## Inference & Submission

The function predict_image(image_path, model) loads each test image, applies the same transforms, and outputs the predicted class.

Results are aggregated into a pandas DataFrame and saved as submission.csv.

