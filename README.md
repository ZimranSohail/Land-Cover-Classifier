# Satellite-Based Land-Cover Classification Using Transfer Learning

A deep learning project that classifies satellite imagery into five
land-cover categories using ResNet-18, transfer learning, and a
Streamlit web application.

## Overview

This project uses Sentinel-2 RGB imagery and ESA WorldCover labels to
train and evaluate a five-class land-cover classifier. Three ResNet-18
training strategies were compared: training from scratch, using a frozen
pretrained backbone, and fine-tuning a pretrained model.

The Streamlit application lets users upload an image and view the
predicted class, confidence score, and class probabilities.

## Land-Cover Categories

  WorldCover label   Class
  ------------------ --------------------------
  10                 Tree Cover
  40                 Cropland
  50                 Built-up
  60                 Bare / Sparse Vegetation
  80                 Permanent Water

## Dataset and Preprocessing

Imagery was collected using Sentinel-2 RGB bands: B4 (red), B3 (green),
and B2 (blue). The image patches represent approximately 640 m × 640 m
ground areas. These bands have a native spatial resolution of 10 metres
per pixel; exporting or resizing images to more pixels does not create
additional native spatial detail.

The aligned 128 × 128 dataset contains 2,500 images:

  Split             Images
  ------------ -----------
  Training           1,766
  Validation           367
  Testing              367
  **Total**      **2,500**

The preprocessing workflow includes collecting imagery and labels,
generating RGB patches in Google Earth Engine, organizing images by
class, creating train/validation/test splits, resizing images to 128 ×
128, and normalizing them with ImageNet channel statistics.

## Experiments

Three ResNet-18 configurations were evaluated on the aligned 128 × 128
dataset.

### Experiment 1: ResNet-18 trained from scratch

The model used randomly initialized weights.

-   Best validation accuracy: 92.80%
-   Test accuracy: 95.10% (349/367 correct)

This served as the baseline experiment.

### Experiment 2: Frozen pretrained ResNet-18

A pretrained ResNet-18 backbone was used as a feature extractor while
the classification head was trained for the five classes.

-   Best validation accuracy: 95.37%
-   Test accuracy: 95.37% (350/367 correct)

This tested the value of pretrained image features without updating the
backbone.

### Experiment 3: Fine-tuned pretrained ResNet-18

The pretrained model was fine-tuned on the land-cover dataset.

-   Best validation accuracy: 95.91%
-   Test accuracy: **97.00% (356/367 correct)**

This was the best-performing configuration among the three experiments.

### Experimental comparison

  Experiment                          Best validation accuracy   Test accuracy
  --------------------------------- -------------------------- ---------------
  ResNet-18 trained from scratch                        92.80%          95.10%
  Frozen pretrained ResNet-18                           95.37%          95.37%
  Fine-tuned pretrained ResNet-18                       95.91%      **97.00%**

The results suggest transfer learning improved performance on this
dataset, with fine-tuning achieving the highest test accuracy. The test
set contains 367 images, so the results should be interpreted in the
context of this dataset and do not guarantee similar performance across
all regions, seasons, or image sources.

## Final Model Evaluation

Per-class results for the fine-tuned model:

  Class                             Precision    Recall     F1-score
  ------------------------------- ----------- --------- ------------
  Bare / Sparse Vegetation             97.50%   100.00%       98.73%
  Built-up                             98.55%   100.00%       99.27%
  Cropland                             91.78%    95.71%       93.71%
  Permanent Water                      98.73%    97.50%       98.11%
  Tree Cover                           98.48%    91.55%       94.89%
  **Macro average F1-score**                              **96.94%**
  **Weighted average F1-score**                           **97.00%**

Cropland and tree cover were comparatively more challenging classes, in
part because some landscapes contain mixtures of fields, vegetation, and
trees.

## Streamlit Application

The application supports image upload and displays the predicted class,
confidence, and class probabilities.

### Application screenshot

Save a screenshot of the app interface as
`screenshots/app_interface.png`.

![Streamlit application interface](screenshots/app_interface.png)

### Example: Lake image prediction

Save the lake prediction screenshot as
`screenshots/lake_prediction.png`.

![Lake image prediction example](screenshots/lake_prediction.png)

The confidence displayed by the application is the model's predicted
probability for a class, not a guarantee that the prediction is correct.
The model was trained on RGB images generated from Sentinel-2 imagery
using a specific visualization and scaling procedure. Google Earth
screenshots may differ in color, scale, acquisition date, and image
characteristics, so such predictions are demonstrations rather than
formal accuracy evaluations.

## Running the Application Locally

### Requirements

-   Python
-   pip
-   The exported model weights and configuration files

### 1. Clone the repository

``` bash
git clone https://github.com/ZimranSohail/Land-Cover-Classifier.git
cd Land-Cover-Classifier
```

### 2. Create a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Linux or macOS:**

``` bash
python -m venv .venv
source .venv/bin/activate
```

If PowerShell blocks activation, use Command Prompt or run the
environment's Python executable directly.

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run Streamlit

``` bash
streamlit run app.py
```

The app should open in your browser. If it does not, use the local URL
printed in the terminal, usually `http://localhost:8501`.

## Repository Structure

``` text
Land-Cover-Classifier/
├── Dataset_generation.ipynb
├── Land_Cover_classifier.ipynb
├── app.py
├── inference.py
├── config.json
├── model_weights.pth
├── requirements.txt
├── README.md
├── .gitignore
├── sample_images/
│   ├── bare_sparse_vegetation.png
│   ├── built_up.png
│   ├── cropland.png
│   ├── permanent_water.png
│   └── tree_cover.png
└── screenshots/
    ├── app_interface.png
    └── lake_prediction.png
```

## Repository Contents

  -----------------------------------------------------------------------
  File                                Purpose
  ----------------------------------- -----------------------------------
  `Dataset_generation.ipynb`          Satellite dataset generation
                                      workflow

  `Land_Cover_classifier.ipynb`       Model training and evaluation
                                      experiments

  `app.py`                            Streamlit user interface

  `inference.py`                      Model loading, preprocessing, and
                                      prediction

  `config.json`                       Model configuration, class mapping,
                                      and normalization

  `model_weights.pth`                 Exported fine-tuned model weights

  `requirements.txt`                  Python dependencies

  `sample_images/`                    Example images for testing the app

  `screenshots/`                      Images embedded in this README
  -----------------------------------------------------------------------

## Limitations

-   The test set contains 367 images; broader evaluation on
    geographically diverse datasets is needed.
-   Some classes have visually similar or mixed boundaries.
-   Acquisition dates and seasonal changes can affect image appearance.
-   The model expects images processed similarly to its training data.
-   Predictions on unrelated screenshots may be unreliable even when
    confidence is high.
-   Test metrics apply to the defined test set, not every real-world
    image.

## Future Improvements

-   Deploy the Streamlit application publicly.
-   Add a confusion matrix and more detailed error analysis.
-   Improve dataset diversity across geographic locations and seasons.
-   Compare additional pretrained architectures.
-   Build a FastAPI endpoint for programmatic predictions.
-   Add automated tests for preprocessing and inference.
-   Evaluate generalization on an independent external dataset.

## Technologies Used

Python · PyTorch · Torchvision · ResNet-18 · Transfer Learning · Google
Earth Engine · Sentinel-2 · ESA WorldCover · Streamlit · Pillow

## Author

**Zimran Sohail**

A deep-learning and remote-sensing project combining satellite image
processing, transfer learning, model evaluation, and application
development.
