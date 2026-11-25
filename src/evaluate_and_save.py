from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, recall_score, precision_score

Path("outputs").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/processed/attrition_clean.csv")

df_model = df.copy()
df_model["Promotion_Lag_Ratio"] = df_model["YearsSinceLastPromotion"] / (
    df_model["YearsAtCompany"] + 1
)
df_model["Tenure_Per_Company"] = df_model["TotalWorkingYears"] / (
    df_model["NumCompaniesWorked"] + 1
)

X = pd.get_dummies(
    df_model.drop(columns=["Attrition", "Attrition_Flag"]), drop_first=True
)
y = df_model["Attrition_Flag"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

lr_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
)
rf_model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
hgb_model = HistGradientBoostingClassifier(random_state=42)

models = {
    "Logistic Regression": lr_pipe,
    "Random Forest": rf_model,
    "HistGradientBoosting": hgb_model,
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)
    auc = roc_auc_score(y_test, probs)
    results[name] = {
        "roc_auc": round(float(auc), 4),
        "quit_precision": round(float(precision_score(y_test, preds, pos_label=1)), 4),
        "quit_recall": round(float(recall_score(y_test, preds, pos_label=1)), 4),
    }
    print(f"\n=== {name} (ROC-AUC: {auc:.3f}) ===")
    print(classification_report(y_test, preds, target_names=["Stayed", "Quit"]))

lr_probs = lr_pipe.predict_proba(X_test)[:, 1]
print("\n--- Logistic Regression Threshold Check ---")
for thresh in [0.50, 0.40, 0.35]:
    t_preds = (lr_probs >= thresh).astype(int)
    p = precision_score(y_test, t_preds, pos_label=1, zero_division=0)
    r = recall_score(y_test, t_preds, pos_label=1, zero_division=0)
    print(f"Threshold {thresh:.2f} -> Quit Precision: {p:.2f}, Quit Recall: {r:.2f}")

joblib.dump(lr_pipe, "outputs/logistic_regression_attrition_model.joblib")
with open("outputs/model_metrics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved model to outputs/logistic_regression_attrition_model.joblib and metrics to outputs/model_metrics.json")