# Violence Detection - Hockey Fight

## Overview
This project aims to develop a machine learning model for detecting violent incidents, specifically fights, in hockey games. Using video analysis, the model identifies and classifies violent actions to enhance safety and provide insights into player behavior during matches.


![](https://github.com/harshakalluri1403/Violence-Detection/blob/4f3363e01dc1289cdd6840fbdde6b3706ba8aa4a/download%20(2).png)

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Jupyter Notebook](#jupyter-notebook)
- [Model Training](#model-training)
- [Evaluation](#evaluation)

## Features
- Detection of fights in hockey games using video analysis.
- Processing of multiple video files (1,000 videos in total).
- Visualization of detected incidents with bounding boxes and annotations.

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/violence-detection-hockey-fight.git
   cd violence-detection-hockey-fight
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```


## Usage
To run the Jupyter notebook, follow these steps:
1. Start Jupyter Notebook:
   ```
   jupyter notebook
   ```
2. Open the violence-detection.ipynb file in your web browser.
3. Follow the instructions in the notebook to load the video data from the data folder and perform analysis.

## Dataset
The dataset used for training and evaluation consists of 1,000 video clips of hockey games, specifically focusing on fight scenarios. The videos are located in the data folder.

## Jupyter Notebook
The main analysis and model training are conducted in the violence-detection.ipynb Jupyter notebook. This notebook includes:
- Data loading and preprocessing.
- Model training.
- Visualization of detection results.
- Evaluation of model performance.
## Model Training
The model can be trained directly within the Jupyter notebook. Ensure that you have set the correct path to the data folder in the notebook cells.

## Evaluation
Model performance can be evaluated using metrics such as precision, recall, F1 score, and a confusion matrix, which are also included in the notebook.


![](https://github.com/harshakalluri1403/Violence-Detection/blob/4f3363e01dc1289cdd6840fbdde6b3706ba8aa4a/download%20(1).png)
![](https://github.com/harshakalluri1403/Violence-Detection/blob/4f3363e01dc1289cdd6840fbdde6b3706ba8aa4a/download.png)
