# COVID-19-Outcome-Prediction
This repository contains the implementation of a machine learning pipeline to predict recovery or death outcomes for individuals based on 14 critical features. The project involves designing and evaluating multiple classifiers to determine the most effective model for this binary classification task.
## Features
Dataset: Includes 14 variables such as demographic, travel history, symptoms, and outcomes (death or recovered).

**Machine Learning Models:**
- K-Nearest Neighbors (KNN)
- Logistic Regression
- Naïve Bayes
- Decision Trees
- Support Vector Machines (SVM)
- Hyperparameter Optimization: Grid search is used to optimize model parameters for best performance.
- Performance Metrics: Models are evaluated based on:
- Precision, Recall, F1-score
- ROC/AUC curves
  
Data Splitting: Dataset is divided into training, validation, and testing sets using stratified sampling to maintain class balance.

Preprocessing: Includes encoding categorical variables, standardizing numerical features, and handling potential class imbalance.
