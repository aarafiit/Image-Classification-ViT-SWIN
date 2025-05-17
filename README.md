# Chest X-Ray Classification using Transformers

This project implements a web application for classifying chest X-ray images using Vision Transformer (ViT) and Swin Transformer models. The application can detect 14 different radiological findings in chest X-ray DICOM images.

## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your trained models in the project directory:
- `vit_model.pth` for the Vision Transformer model
- `swin_model.pth` for the Swin Transformer model

## Running the Application

Run the Streamlit application:
```bash
streamlit run app.py
```

The application will open in your default web browser.

## Usage

1. Select the model you want to use (ViT or Swin Transformer)
2. Upload a DICOM format chest X-ray image
3. View the predictions and probability scores for each finding

## Supported Findings

The model can detect the following radiological findings:
- Aortic enlargement
- Atelectasis
- Calcification
- Cardiomegaly
- Consolidation
- ILD
- Infiltration
- Lung Opacity
- Nodule/Mass
- Other lesion
- Pleural effusion
- Pleural thickening
- Pneumothorax
- Pulmonary fibrosis
- No finding



Notebook Link : https://www.kaggle.com/code/abdullahalrafi143/finalcodebase
