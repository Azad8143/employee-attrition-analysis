import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

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

models = {
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print(f"\n=== {name} (ROC-AUC: {roc_auc_score(y_test, probs):.3f}) ===")
    print(classification_report(y_test, preds, target_names=["Stayed", "Quit"]))

rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(
    ascending=False
)
print("\nTop 10 Attrition Predictors:\n", importances.head(10).round(4))