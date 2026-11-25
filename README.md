# Employee Attrition Prediction System

End-to-end machine learning pipeline analyzing employee turnover drivers and predicting attrition risk on the IBM HR Analytics dataset (1,470 employees).

## Key Insights (EDA)
- **Overtime:** Employees working overtime leave at **30.5%** vs. **10.4%** without overtime.
- **Job Role:** **Sales Representatives** have the highest attrition (**39.8%**), followed by **Laboratory Technicians** (**23.9%**) and **Human Resources** (**23.1%**).
- **Compensation & Travel:** Lower monthly income, frequent business travel (**24.9%**), and single marital status (**25.5%**) correlate strongly with turnover.
- **Engineered Features:** `Tenure_Per_Company` (`TotalWorkingYears / (NumCompaniesWorked + 1)`) ranked as the #1 feature importance predictor.

## Model Performance (20% Stratified Test Split)
| Model | ROC-AUC | Quit Precision | Quit Recall | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (`class_weight="balanced"`, threshold 0.40)** | **0.794** | **0.35** | **0.77** | — |
| Logistic Regression (default threshold 0.50) | 0.794 | 0.35 | 0.64 | 76% |
| Random Forest (`n_estimators=200`, balanced) | 0.749 | 0.42 | 0.28 | 82% |
| HistGradientBoosting | 0.733 | 0.62 | 0.17 | 85% |

Logistic Regression tuned to a **0.40** probability threshold was selected as the final model because it catches **77%** of actual leavers while preserving precision (`0.35`) and achieving the highest **ROC-AUC (0.794)**.

## Project Structure
```text
employee-attrition-analysis/
├── data/
│   ├── raw/WA_Fn-UseC_-HR-Employee-Attrition.csv
│   └── processed/attrition_clean.csv
├── notebooks/
│   └── 01_attrition_analysis.ipynb
├── outputs/
│   ├── figures/key_attrition_drivers.png
│   ├── logistic_regression_attrition_model.joblib
│   ├── model_metrics.json
│   └── scored_employees.csv
├── src/
│   ├── preprocessing.py
│   ├── eda.py
│   ├── train_model.py
│   ├── evaluate_and_save.py
│   └── predict_attrition.py
└── requirements.txt