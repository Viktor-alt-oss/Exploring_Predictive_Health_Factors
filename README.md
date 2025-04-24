# Predicting PCOS Risk Through Lifestyle Factors

## Project Overview
This project develops a machine learning model to predict Polycystic Ovary Syndrome (PCOS) risk based on lifestyle and health metrics. PCOS is a complex endocrine disorder affecting 6-12% of reproductive-aged women worldwide, with significant impacts on metabolic health and fertility. Our solution analyzes key modifiable lifestyle factors to enable early risk identification and personalized intervention strategies.

**Competition:** [Exploring Predictive Health Factors](https://www.kaggle.com/c/exploring-predictive-health-factors/leaderboard)

## Dataset Description
The dataset contains anonymized health records of women with and without PCOS diagnosis, featuring:

- **210 training samples** with complete diagnostic labels
- **13 predictive features** across key domains:
  - **Demographics**: Age categories
  - **Clinical markers**: Weight, hormonal imbalance indicators
  - **Lifestyle factors**: Exercise patterns, sleep duration, activity benefits
  - **PCOS symptoms**: Hirsutism, insulin resistance, conception difficulty

Data was synthetically generated from real clinical distributions to preserve statistical relationships while ensuring privacy. Significant feature engineering was required to harmonize categorical variables across multiple inconsistent encodings.

## Methodology

### Key Data Processing Steps:
1. **Value Standardization**:
   - Consolidated 11 age categories → 5 standardized ranges
   - Normalized 8 exercise type descriptions → 4 core categories
   - Harmonized inconsistent benefit/duration scales

2. **Missing Value Treatment**:
   - Used modal imputation for categorical features
   - Addressed 23 incomplete records (11% of dataset)

3. **Feature Encoding**:
   - One-hot encoding for all categorical variables
   - Binary conversion of target variable (PCOS: Yes/No → 1/0)

### Model Development:
- Evaluated 7 classification algorithms:
  - Logistic Regression (Accuracy: 78.6%)
  - SVM (78.6%)
  - Decision Trees (78.6%)
  - **Random Forest (83.3%)** ← Selected as baseline
  - Gradient Boosting (83.3%)
  - AdaBoost (76.2%)
  - K-Nearest Neighbors (76.2%)

- Optimized Random Forest via GridSearchCV:
  - Tested 100 hyperparameter combinations
  - Achieved **85.7% accuracy** (14.3% error rate)
  - Best parameters: 56 estimators, 6 max features

## Key Results

### Performance Metrics:
- **Accuracy**: 85.7% 
- **Precision**:
  - PCOS detection: 67%
  - Non-PCOS identification: 86%
- **Recall**:
  - PCOS cases: 44%
  - Non-PCOS cases: 94%
- **ROC AUC**: 0.81

### Clinical Insights:
- Exercise frequency and type showed strongest predictive power
- Sleep duration emerged as significant secondary factor
- Hormonal markers provided important diagnostic confirmation

## How to Use
from joblib import load

model = load('Exploring_Predictive_Health_Factors.joblib')

# Prepare new data (must match training schema)
predictions = model.predict(new_data)
### Requirements:
```bash
pip install pandas scikit-learn numpy matplotlib seaborn joblib

