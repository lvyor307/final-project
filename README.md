# Stuttering Type Prediction

## Overview

This project aims to develop a model for predicting the types of stuttering exhibited in speech samples. The task involves multiclass classification, where the model categorizes speech segments into different types of stuttering events.

## Dataset

The dataset used for training and evaluation is the Kassel State of Fluency Dataset (KSoF). This dataset contains labeled speech samples from individuals with stuttering, recorded during therapy sessions at the Institut der Kasseler Stottertherapie. The dataset comprises various types of stuttering events, including blocks, prolongations, sound repetitions, word repetitions, interjections, and speech modifications specific to therapy sessions.

## Approach

### 1. Feature Extraction

The project explores the use of feature extraction techniques to capture relevant information from the speech samples. Features are extracted using open-source libraries such as audeep, deepspectrum, opensmile, and xbow. Although the exact workings of these libraries in feature extraction are yet to be fully comprehended, initial experiments suggest that they contribute significantly to the richness of the dataset.

### 2. Model Development

Two approaches are considered for model development:

- **With Feature Extraction:** In this approach, the extracted features are used as input to train the predictive model. Various machine learning algorithms will be explored to develop an effective classifier for stuttering type prediction.

- **Without Feature Extraction:** As an alternative approach, the raw speech samples are utilized directly to train the model. This method involves preprocessing the audio data, such as normalization and segmentation, followed by feature engineering to represent the speech signals effectively. The same set of machine learning algorithms will be applied to compare the performance against the feature-extracted approach.

## Evaluation

The performance of the models will be evaluated using standard metrics for multiclass classification tasks, including accuracy, precision, recall, and F1-score. Cross-validation techniques will be employed to ensure robustness and generalization of the models.

## Results and Analysis

Upon completion of the experiments, a thorough analysis will be conducted to compare the performance of the models with and without feature extraction. Insights gained from this analysis will help determine the effectiveness of the extracted features in improving the model's predictive capability for stuttering type prediction.

## Requirements

- Python 3.11
- Required Python libraries (will be updates during the project development)


## Conclusion

The README will be updated with the project's findings, conclusions, and recommendations based on the experimental results.
