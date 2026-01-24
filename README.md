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

### 📊 Scatter Plot: outlier detection 
Shows relationship between features with outliers and target variable.

<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/scatter.png" width="900"/>

### 🎯 Target Transformation 
Before and after log transformation of SalePrice.

| Before | After |
|--------|-------|
| <img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/target1.png" width="450"/> | <img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/target2.png" width="450"/> |


### 🔥 Feature Correlation Heatmap  
After dropping highly correlated features (+0.80).

<img src="https://github.com/radwanhefny/House-Price-Advanced-Regression-Techniques/blob/main/pictures/heatmap.png" width="900"/>

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
  - submission.csv → Kaggle submission.
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
2. Clean missing values and outliers.
3. Apply feature engineering and encoding:
    - Ordinal encoding
    - Target encoding
    - Log transformation of target
    - New engineered features
4. Split train/validation sets.
5. Build preprocessing pipeline (numerical scaling, categorical encoding).
6. Train ensemble of Ridge, CatBoost, XGBoost, LightGBM, Gradient Boosting.
7. Tune hyperparameters using Optuna for stability and performance.
8. Generate Kaggle-ready predictions.

### 🧪 Experimental Notes
- Tested Random Forest: underperformed compared to ensemble.  
- Tried manual ensemble: similar performance to Voting, but Voting automated the process.  
- Explored stacking in multiple versions: Voting consistently gave better results, so it was chosen.  
- Added Linear Regression in the final version: overfitting occurred, reducing ensemble performance due to sensitivity to extreme values.

---

## 🤝 Contributing
Contributions are welcome!
1. Fork the repository
2. Create a new feature branch
3. Submit a pull request
Please ensure your code is clean, structured, and well-commented.


---


## 📝 License
This project is licensed under the MIT license - see the LICENSE file for details. 


---


## 📞 Support
If you have questions or need help, feel free to:
- Open an issue on this repository  
- Connect with me on LinkedIn: https://www.linkedin.com/in/radwanhefny  
