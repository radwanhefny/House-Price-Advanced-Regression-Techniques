# 🏠 Ames Housing Price Prediction | Advanced Regression Techniques

This project implements an **end-to-end machine learning pipeline** for predicting house prices on the Ames Housing dataset. It combines manual ordinal encoding, target encoding, robust preprocessing, and a weighted ensemble of Ridge, CatBoost, XGBoost, LightGBM, and Gradient Boosting Regressor, with **hyperparameter tuning using Optuna**, optimized for performance and generalization.

Kaggle Public Score: **0.12260 | Top 12% (Rank 586 / 4764)**

---

## ✨ Features
- Robust **data cleaning** and **EDA-driven outlier removal**.
- Advanced **feature engineering**:
  - Manual ordinal encoding
  - Target encoding
  - New features: house age, total area, total baths, total porch area
- **Pipeline-based preprocessing** for numerical and categorical features.
- Custom **weighted ensemble** combining multiple regressors.
- **Hyperparameter tuning** for all models using **Optuna**.
- Generates Kaggle-ready submission automatically.

---

## 📋 Prerequisites
- Python 3.8+
- Libraries: `numpy`, `pandas`, `scikit-learn`, `xgboost`, `catboost`, `lightgbm`, `category_encoders`
- Dataset: Place `train.csv` and `test.csv` inside a `data/` folder

---

## 🚀 Getting Started
1. Clone the repository:
```bash
git clone https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques.git
cd House-Price-Advanced-Regression-Techniques
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
3. Run the pipeline script:
```bash
python house-prices-advanced-regression-techniques.py
```
This will preprocess the data, train the ensemble, and generate submission.csv.

---

## 🎬 Screenshots / Demo

### 📊 Scatter Plot: Engineered Features 
Shows relationship between new features and target variable.
<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/scatter.png" width="500"/>

### 🎯 Target Transformation 
Before and after log transformation of SalePrice.
**Before**
<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/target1.png" width="500"/>
**After**
<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/target2.png" width="500"/>

### 🔥 Feature Correlation Heatmap  
After dropping highly correlated features.
<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/heatmap.png" width="500"/>

---

## 🗂️ Project Structure
```
📁 House-Price-Advanced-Regression-Techniques
├── house-prices-advanced-regression-techniques.py   # Full ML pipeline
├── house-prices-advanced-regression-techniques.ipynb   # Full ML jupyter notebook
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── pictures/                                        # Visualizations
│   ├── scatter.png
│   ├── target1.png
│   ├── target2.png
│   └── heatmap.png
├── submission.csv                                # Kaggle submission
├── hyperparameter_optimization.ipynb
├── requirements.txt
└── README.md
```
---

## 🛠️ Usage
- Run the pipeline script to generate preprocessed data, train models with tuned hyperparameters, and produce Kaggle-ready predictions.
- Outputs:
  - submission_v4.csv → Kaggle submission.
  - Internal validation RMSE printed in console.
- Expected Kaggle Public Score: 0.12260 | Top 12%.

---

## ✅ Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Cross-validation RMSE
- Kaggle Public Score

---

## 🧠 How It Works
1. Load datasets using Pandas.
2. Separates X (size, bedrooms) and y (price).
3. Normalizes features manually or using standardization.
4. Adds a column of ones for the bias term.
5. Implements the hypothesis function.
6. Implements Cost Function.
7. Implements Gradient Descent (vectorized).
8. Updates parameters until convergence.
9. Plots the cost function to visualize learning progress.
