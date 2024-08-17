
# Logistic Regression Model on Sample Dataset

This project demonstrates the implementation of a logistic regression model using a simple dataset. The goal is to classify data points based on two features (`X1` and `X2`) and evaluate the model's performance.

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Data Overview](#data-overview)
- [Data Visualization](#data-visualization)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Analysis and Inference](#analysis-and-inference)

## Installation

To run the project, ensure you have Python installed along with the required libraries. You can install the necessary libraries using:

```bash
pip install -r requirements.txt
```

## Project Structure

The project files are organized as follows:

```plaintext
├── sample_data_analysis.ipynb   # Jupyter notebook with data analysis and model training
├── README.md                    # Project documentation
├── requirements.txt             # List of required libraries
```

## Data Overview

The dataset used in this project contains 100 observations with two features (`X1`, `X2`) and a binary target variable (`y`):

```plaintext
X1         | X2         | y
-----------|------------|---
-2.988372  | 8.828627   | 0
5.722930   | 3.026972   | 1
-3.053580  | 9.125209   | 0
...        | ...        | ...
-2.504084  | 8.779699   | 0
```

The target variable `y` is binary, indicating the class to which each observation belongs (0 or 1).

## Data Visualization

To understand the distribution of the data, a scatter plot is generated:

```python
data.plot.scatter("X1", "X2", c="y", colormap="winter")
```

This visualization helps in identifying the separability of the two classes based on the features `X1` and `X2`.

## Model Training

A logistic regression model is employed to classify the data points. The process involves:

1. **Splitting the Data**: The dataset is divided into training and testing sets.
2. **Training the Model**: The logistic regression model is trained on the training data using the `lbfgs` solver.
3. **Making Predictions**: The model generates predictions on the test data.

Here’s the code snippet for model training:

```python
log_classifier = LogisticRegression(solver="lbfgs", random_state=1, max_iter=200)
log_classifier.fit(X_train, y_train)
```

## Evaluation

The model's performance is evaluated using the accuracy score metric, which compares the predicted labels with the true labels in the test set:

```python
from sklearn.metrics import accuracy_score

y_pred = log_classifier.predict(X_test)
print(f"Logistic regression model accuracy: {accuracy_score(y_test, y_pred):.3f}")
```

The accuracy score obtained from the model is:

```plaintext
Logistic regression model accuracy: 1.000
```

## Analysis and Inference

### Analysis

The logistic regression model achieved an accuracy of 100% on the test data. This means that the model correctly classified all the test data points into their respective classes (`0` or `1`). The high accuracy can be attributed to the following factors:

1. **Linear Separability**: The scatter plot indicates that the two classes are likely linearly separable. Logistic regression, which is a linear classifier, performs well when classes are linearly separable.

2. **Model Simplicity**: The logistic regression model is a simple yet effective classifier for binary classification tasks, particularly when the dataset is small and the decision boundary is linear.

### Inference

Given the perfect accuracy score, the following inferences can be drawn:

- **High Confidence**: The model's performance suggests that it has learned the underlying pattern in the data very well, and it is likely to generalize well on similar unseen data.

- **Potential Overfitting**: While a 100% accuracy is impressive, it is essential to be cautious. Such a high score may indicate that the model is overfitting to the training data, especially if the dataset is small or the decision boundary is overly simplistic.

- **Dataset Suitability**: The dataset's features (`X1` and `X2`) appear to be well-suited for classification using logistic regression, further validating the model's performance.

In real-world applications, it is crucial to validate the model on different datasets and possibly introduce cross-validation to ensure that the model is not overfitting and is genuinely performing well.
