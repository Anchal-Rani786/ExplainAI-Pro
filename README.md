![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-orange)
![ExplainableAI](https://img.shields.io/badge/ExplainableAI-LIME-red)
![MLOps](https://img.shields.io/badge/MLOps-MLflow-green)

# ExplainAI-Pro


**ExplainAI-Pro is an Explainable AI framework for interpreting deep learning image classification models.**

The system uses a pretrained ResNet18 CNN model and generates explanations for predictions using LIME, while tracking experiments using MLflow.

---

## Features

• Deep Learning image classification using PyTorch
• Explainable AI interpretation using LIME
• Visualization dashboard highlighting important image regions
• Prediction confidence logging
• Experiment tracking using MLflow (MLOps)

---

## System Pipeline

Input Image → CNN Model (ResNet18) → Prediction → LIME Explanation → Visualization Dashboard → MLflow Experiment Tracking

---

## Technologies Used

Python
PyTorch
Computer Vision
Explainable AI (LIME)
MLflow (MLOps)

---

## Project Structure

ExplainAI-Pro
│
├── images
│   └── test.jpg
│
├── outputs
│   ├── lime_result.png
│   ├── comparison.png
│   └── prediction.json
│
├── src
│   ├── main.py
│   ├── model.py
│   ├── predict.py
│   ├── explain_lime.py
│   ├── visualization.py
│   └── mlflow_tracker.py
│
├── requirements.txt
└── README.md

---

## Run Project

pip install -r requirements.txt

python src/main.py

---

## Output

• Model prediction with confidence score
• LIME explanation heatmap
• Visualization comparison dashboard
• MLflow experiment tracking

## Results

### LIME Explanation

![LIME](outputs/lime_result.png)

### Visualization Dashboard

![Dashboard](outputs/comparison.png)
