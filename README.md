# Predicting Employee Attrition Using Machine Learning

## Overview

This project implements several machine learning models to predict employee attrition (binary classification) using the IBM HR Analytics Employee Attrition dataset. The work follows the structure of data preparation, preprocessing, modeling, hyperparameter tuning, and evaluation.

## Dataset

- Source: IBM HR Analytics Employee Attrition & Performance
- Size: 1,470 records, 35 features
- Target: Attrition (Yes/No)

## Methodology

### Data Preparation
- Removed redundant columns: EmployeeCount, EmployeeNumber
- One-hot encoding for categorical features (Gender, OverTime, MaritalStatus, Department, EducationField, JobRole)
- Ordinal mapping for BusinessTravel

### Preprocessing
- Standard scaling of numerical features
- Train-test split

### Models
- Logistic Regression (L2 penalty, GridSearchCV on regularization parameter C)
- K-Nearest Neighbors
- Gradient Boosting Classifier
- Additional models explored: Decision Tree, Random Forest, AdaBoost, Naive Bayes

### Evaluation
- Primary metric: AUC-ROC (10-fold cross-validation)
- Secondary metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix and full classification report

## Results

| Model                        | AUC    | Accuracy | Precision (No Attrition) | Recall (No Attrition) |
|------------------------------|--------|----------|---------------------------|-----------------------|
| Logistic Regression (tuned)  | 0.8335 | 0.698    | 0.916                     | 0.716                 |
| Gradient Boosting Classifier | -      | 0.863    | 0.880                     | 0.980                 |
| K-Nearest Neighbors          | -      | 0.780    | -                         | -                     |

The tuned Logistic Regression model achieved the highest AUC.

## Requirements and Execution

```bash
git clone https://github.com/yourusername/employee-attrition-prediction.git
cd employee-attrition-prediction
pip install -r requirements.txt
jupyter notebook ISOM_3360_Project_Coding_File_Group_5.ipynb
```

All cells are executable sequentially from top to bottom.

## Contributors

Li Weitao (Wade) – 20802600
Li Kexin (Crystal) – 20731289
Talha Hameed Khan – 20850023

## References

Kaggle Notebook: https://www.kaggle.com/code/stanley0010/isom3360project

Dataset: https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset
