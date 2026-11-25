from pathlib import Path
import joblib
import pandas as pd

model = joblib.load("outputs/logistic_regression_attrition_model.joblib")
df = pd.read_csv("data/processed/attrition_clean.csv")

df_feat = df.copy()
df_feat["Promotion_Lag_Ratio"] = df_feat["YearsSinceLastPromotion"] / (
    df_feat["YearsAtCompany"] + 1
)
df_feat["Tenure_Per_Company"] = df_feat["TotalWorkingYears"] / (
    df_feat["NumCompaniesWorked"] + 1
)

X = pd.get_dummies(
    df_feat.drop(columns=["Attrition", "Attrition_Flag"], errors="ignore"),
    drop_first=True,
)
X = X.reindex(columns=model.feature_names_in_, fill_value=0)

probs = model.predict_proba(X)[:, 1]
threshold = 0.40

scored = df[["Attrition", "OverTime", "JobRole", "MonthlyIncome"]].copy()
scored["Attrition_Risk_Prob"] = probs.round(3)
scored["Predicted_High_Risk"] = (probs >= threshold).astype(int)

Path("outputs").mkdir(parents=True, exist_ok=True)
scored.sort_values("Attrition_Risk_Prob", ascending=False).to_csv(
    "outputs/scored_employees.csv", index=False
)

print("Top 10 Highest-Risk Employees:")
print(scored.sort_values("Attrition_Risk_Prob", ascending=False).head(10).to_string(index=False))
print(f"\nFlagged {scored['Predicted_High_Risk'].sum()} of {len(scored)} employees at threshold {threshold:.2f}.")