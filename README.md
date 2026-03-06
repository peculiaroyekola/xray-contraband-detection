# AI Contraband Detection in X-Ray Images

Deep learning object detection system developed to identify contraband items in full-body X-ray scans using Faster R-CNN.

This project was developed during the **Computer Vision & Data Science Minor at NHL Stenden University of Applied Sciences**.

---

## Project Overview

Security scanners used in airports and checkpoints produce full-body X-ray images that may contain hidden contraband objects.  
This project investigates how deep learning models can automatically detect such objects.

The system was implemented using **Faster R-CNN with a ResNet-50 backbone** trained on a dataset of X-ray scans.

Dataset:

- 451 full-body X-ray scans
- 33 annotated object classes

---

## Model Architecture

The object detection model used:

Faster R-CNN  
ResNet-50 Backbone  
Feature Pyramid Network (FPN)

The pipeline includes:

- data preprocessing
- augmentation experiments
- model training
- validation
- testing

---

## Experiments

Different preprocessing and augmentation methods were evaluated:

- CLAHE
- Gamma correction
- Emboss filtering
- Gaussian blur

The goal was to evaluate how image preprocessing affects small object detection in X-ray scans.

---

## Results

Best results achieved:

| Object Class | F1 Score |
|---------------|---------|
| Surgical Implant | 0.812 |
| Zipper | 0.595 |
| Dental | 0.437 |

These results were obtained using the baseline preprocessing configuration. :contentReference[oaicite:1]{index=1}

---

## Repository Contents

Technical Paper  
Poster Presentation  
Experiment Results  
Detection Prototype  
Model Training Pipeline

---

## Technologies Used

Python  
PyTorch  
Deep Learning  
Computer Vision  
TensorBoard  
Albumentations
