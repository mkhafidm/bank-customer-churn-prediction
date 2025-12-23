# Bank Customer Churn Prediction

This project focuses on building an end-to-end machine learning pipeline to predict customer churn in the banking industry. 
The model aims to identify customers who are likely to leave the bank, enabling proactive retention strategies.

---

## Objective
The primary objective of this project is to predict customer churn using machine learning techniques, with a focus on 
maximizing recall for churn customers while maintaining a reasonable balance with precision.

---

## Dataset
The dataset used in this project is a publicly available bank customer churn dataset obtained from Kaggle for educational and research purposes.
Dataset source: https://www.kaggle.com/datasets/gauravtopre/bank-customer-churn-dataset/
- Total records: 10,000 customers
- Target variable: `churn` (1 = churn, 0 = non-churn)
- Features include demographic, financial, and behavioral attributes
- The dataset exhibits class imbalance, which is addressed during modeling

---

## Methodology
1. Data cleaning and preprocessing  
2. Feature encoding and train-test split with stratification  
3. Handling class imbalance using `scale_pos_weight`  
4. Model training using XGBoost  
5. Hyperparameter tuning with Optuna  
6. Threshold optimization  
7. Model interpretability using SHAP  

---

## Results
- ROC-AUC: ~0.87 (optimized model)
- Improved recall for churn customers after hyperparameter tuning
- Threshold tuning allows flexible precision–recall trade-off
- SHAP analysis identifies key churn drivers such as age, number of products, and active membership status

---

## Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Optuna
- SHAP
- Matplotlib, Seaborn

---

## How to Run
```bash
pip install -r requirements.txt
```

---

## Key Takeaways
- Handling class imbalance and threshold tuning significantly improves churn detection performance
- XGBoost combined with Optuna provides strong predictive capability for tabular churn data
- Threshold selection plays a critical role in aligning model performance with business objectives
- SHAP enhances model transparency by revealing key drivers of customer churn

