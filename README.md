# Supervised Learning Project

##  Overview

This project focuses on implementing **Supervised Machine Learning** techniques using Python and its data science and machine learning libraries. It covers two major types of supervised learning: **Regression and Classification**.

The project involves working with real-world datasets, performing Exploratory Data Analysis (EDA), preprocessing data, training machine learning models, and evaluating their performance using appropriate evaluation metrics.

Two different datasets are used to understand how supervised learning algorithms solve regression and classification problems.

##  Objectives

- Understand the fundamentals of supervised machine learning.
- Perform Exploratory Data Analysis (EDA) on real-world datasets.
- Apply data preprocessing techniques to prepare data for machine learning.
- Implement regression and classification algorithms.
- Train models using training datasets and generate predictions on test data.
- Evaluate model performance using suitable evaluation metrics.

##  Technologies Used

- **Python** – Programming language used for implementing the project.
- **Pandas** – Data loading, manipulation, and analysis.
- **Matplotlib** – Data visualization.
- **Seaborn** – Statistical data visualization.
- **Scikit-learn** – Machine learning algorithms, preprocessing, data splitting, and model evaluation.

##  Concepts Used

- Supervised Machine Learning
- Regression and Classification
- Data Loading and Inspection
- Exploratory Data Analysis (EDA)
- Descriptive Statistics
- Data Visualization
- Missing Value Handling
- Label Encoding
- One-Hot Encoding
- Feature Scaling (StandardScaler)
- Train-Test Split
- Linear Regression
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Model Training and Prediction
- Model Evaluation
- Mean Squared Error (MSE)
- Classification Report
- Confusion Matrix

---

#  Datasets

This project uses two datasets for different supervised learning tasks.

### California Housing Dataset

Used for the Regression task.

The dataset is fetched directly using Scikit-learn.

**Selected Features:**
- MedInc – Median Income
- HouseAge – Median House Age
- AveRooms – Average Number of Rooms

**Target Variable:**
- MedHouseVal – Median House Value

### Telco Customer Churn Dataset

Used for the Classification task.

The dataset contains customer information from a telecommunications company.

**Objective:** Predict whether a customer will leave the telecom service based on available customer-related features.

**Target Variable:** `Churn`

**Dataset File:** `WA_Fn-UseC_-Telco-Customer-Churn.csv`

**Dataset Source:** [Download Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

**Note:** The CSV file is not included in this repository. To run the classification script, download the dataset and place the CSV file at the local file path specified in the `pd.read_csv()` statement in the Python script.

---

#  What I Did

## Regression – California Housing Dataset

**Objective:** Predict median house values using selected housing-related features.

### Workflow

- Loaded the California Housing Dataset using Scikit-learn.
- Inspected the dataset using `info()` and `describe()`.
- Visualized relationships between selected variables using Seaborn Pairplot.
- Checked for missing values.
- Selected three independent features: MedInc, HouseAge, and AveRooms.
- Split the dataset into training and testing sets (80:20).
- Trained a Linear Regression model.
- Generated predictions using the test dataset.
- Evaluated model performance using Mean Squared Error (MSE).

### Algorithm Used

**Linear Regression**

### Evaluation Metric

- Mean Squared Error (MSE)

---

## Classification – Telco Customer Churn Dataset

**Objective:** Predict customer churn using customer-related information.

### Workflow

- Loaded the Telco Customer Churn dataset using Pandas.
- Converted `TotalCharges` into numeric format.
- Inspected the dataset using `info()` and `describe()`.
- Visualized customer churn distribution using a Count Plot.
- Handled missing numerical values using mean imputation.
- Encoded the target variable (`Churn`) using Label Encoding.
- Converted categorical features into numerical format using One-Hot Encoding.
- Separated independent features and the target variable.
- Split the dataset into training and testing sets (80:20).
- Applied StandardScaler for feature scaling.
- Trained two classification models.
- Generated predictions and evaluated both models.

### Algorithms Used

**Logistic Regression**
- Used to predict whether a customer will leave the telecom service.

**K-Nearest Neighbors (KNN)**
- Used to classify customers based on their nearest data points.
- Configured with 5 neighbors and distance-based weighting.

### Evaluation Metrics

- Classification Report
- Precision
- Recall
- F1-Score
- Confusion Matrix

---

# Model Evaluation

Different evaluation metrics are used depending on the type of machine learning problem.

| Task | Algorithm | Evaluation Metric |
|---|---|---|
| Regression | Linear Regression | MSE |
| Classification | Logistic Regression | Precision, Recall, F1-Score, Confusion Matrix |
| Classification | KNN | Precision, Recall, F1-Score, Confusion Matrix |

The classification reports and confusion matrices help analyze and compare the predictions made by Logistic Regression and KNN.

---

# Key Observations

- Supervised learning can be applied to both continuous value prediction and categorical prediction problems.
- The California Housing dataset demonstrates how Linear Regression can be used for predicting house values.
- The Telco Customer Churn dataset demonstrates how classification algorithms can identify potential customer churn.
- Feature scaling is applied before training the classification models.
- Encoding techniques make categorical data suitable for machine learning algorithms.
- Different evaluation metrics are required for regression and classification tasks.

---

# Future Scope

- Explore additional features to improve model performance.
- Implement more regression and classification algorithms.
- Perform Cross-Validation for more reliable model evaluation.
- Explore hyperparameter tuning to optimize model performance.
- Compare additional models using suitable evaluation metrics.
- Improve the preprocessing workflow using Scikit-learn Pipelines.
- Develop an interactive dashboard to present model predictions and results.

---

# Author

**Rishit**

Python Developer | Building Projects and Learning by Practice
