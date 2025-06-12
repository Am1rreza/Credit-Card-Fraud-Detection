# Credit Card Fraud Detection 💳🔍

This project aims to detect fraudulent credit card transactions using machine learning. The dataset consists of 284,807 transactions, each described by 30 numerical features: 28 anonymized PCA-transformed variables (V1–V28), the transaction time, and the transaction amount. The goal is to identify the rare fraudulent cases (only ~0.17% of the data) using a supervised learning model. A Random Forest classifier was trained to distinguish between legitimate and fraudulent transactions with high accuracy and strong recall on minority class detection.

## ⚙️ Key Features

- Utilizes a real-world dataset with over 284,000 credit card transactions.
- Handles extreme class imbalance (only ~0.17% of transactions are fraudulent).
- Trains a Random Forest Classifier for binary classification (fraud vs. not fraud).
- Performs exploratory data analysis (EDA) to understand transaction behavior.
- Evaluates model using accuracy, precision, recall, F1-score, and confusion matrix.
- Visualizes fraud vs. non-fraud distribution and amount statistics.

## 📊 Dataset Overview

The dataset contains **284,807** credit card transactions made in two days, with the following features:

- **Time**: Seconds elapsed between each transaction and the first transaction in the dataset.
- **V1–V28**: Principal components obtained from PCA to anonymize original features.
- **Amount**: Transaction amount.
- **Class**: Target label (0 = legitimate, 1 = fraud). Only **492 out of 284,807** transactions are labeled as fraud, making this a highly imbalanced dataset (~0.17% fraud rate).

### 🔍 Class Distribution:
- **Valid transactions**: 284,315  
- **Fraudulent transactions**: 492  
- **Outlier fraction**: 0.00173

## 🧪 Project Workflow

The project follows these main steps:

1. **Data Exploration & Analysis**
   - Printed sample rows to inspect feature values and understand structure.
   - Checked the distribution of the `Class` variable to confirm data imbalance.
   - Analyzed `Amount` statistics separately for fraudulent and valid transactions.

2. **Data Preparation**
   - Separated features (`X`) and target labels (`y`).
   - Performed a stratified train-test split to preserve the class distribution.

3. **Model Training**
   - Used a `RandomForestClassifier` for training due to its robustness and performance on imbalanced datasets.
   - Trained the model on 80% of the data.

4. **Model Evaluation**
   - Evaluated performance using accuracy, precision, recall, and F1-score.
   - Plotted confusion matrix to visualize prediction quality, especially for minority class (fraud).

## 📈 Model Performance & Evaluation

The trained Random Forest model achieved high performance on both training and test sets:

- **Training Accuracy:** 99.99%
- **Test Accuracy:** 99.96%

### 📊 Classification Report (on test set):

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| 0 (Valid) | 1.00 | 1.00 | 1.00 | 56,864 |
| 1 (Fraud) | 0.94 | 0.82 | 0.87 | 98 |

- **Macro Avg F1-score:** 0.94  
- **Weighted Avg F1-score:** 1.00  

💡 Despite the severe class imbalance, the model shows strong recall on the fraud class, which is essential in fraud detection where missing a fraudulent transaction can be costly.

🧱 Confusion matrix and visualizations further confirm the model's ability to correctly detect most fraudulent cases with very few false positives.
