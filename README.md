# AI Classification of Bike Rental Demand – Advanced Topics in AI

This repository contains the coursework submission for the module **6219COMP: Advanced Topics in AI** at Liverpool John Moores University. The project applies machine learning techniques to predict hourly bike rental demand using the **UCI Bike Sharing Dataset**.

---

## Overview

The goal of this project is to classify bike rental demand into three categories: **Low**, **Medium**, and **High**, based on weather and temporal features. It showcases data preprocessing, feature engineering, AI model implementation, evaluation and comparison.

---

## Dataset

- **Source**: [UCI Machine Learning Repository – Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- **Records**: 17,379 hourly entries
- **Features**: Season, temperature, humidity, hour of day, day of week etc.
- **Target**: `cnt` (total rental count) — transformed into a categorical variable (`demand_class`)

---

## AI Techniques

### Primary Model: **Random Forest Classifier**
- Robust to overfitting
- Handles both categorical and numerical features
- Requires minimal preprocessing (e.g., no normalization)
- Performs internal feature selection and provides importance scores

### Compared Model: **Decision Tree**
- Used as a baseline for comparison
- More interpretable, but less accurate due to high variance

---

## Preprocessing

Implemented in `preprocessing.py`:
- Removed leakage-prone features: `casual`, `registered`, `cnt`
- Converted categorical variables to type `category`
- Applied **one-hot encoding**
- Transformed `cnt` into `demand_class` using `pandas.qcut()` to create 3 balanced classes
- Saved two CSV files:
  - `original_bike_data.csv` – cleaned, before encoding
  - `preprocessed_bike_data.csv` – ready for model input

---

## Implementation & Evaluation

Implemented in `model.py` using **scikit-learn**:
- Train-test split (80/20) with class stratification
- Random Forest with **GridSearchCV** for hyperparameter tuning
- Evaluation metrics:
  - **Accuracy**: `86.4%`
  - **F1-Scores**:
    - Low: `0.895`
    - Medium: `0.809`
    - High: `0.892`
  - Confusion matrix shows minimal misclassification between adjacent classes

---

## Model Comparison

| Model           | Accuracy | F1-Score (Medium) |
|----------------|----------|-------------------|
| Random Forest  | 86.4%    | 0.809             |
| Decision Tree  | 80.1%    | 0.726             |

- **Random Forest** handled class overlap better and generalized more effectively.
- **Decision Tree** was faster but more prone to overfitting and less stable.

---

## Repo Structure

```plaintext
6219COMP_AI_Technology/
├── ImplementationSummary.pdf
├── AITechnique/
│   ├── preprocessing.py
│   ├── model.py
│   ├── decision_tree_baseline.py
│   ├── original_bike_data.csv
│   ├── preprocessed_bike_data.csv
│   └── requirements.txt
```

---

## 📎 References

See full list in `ImplementationSummary.pdf`.
****
