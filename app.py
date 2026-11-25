import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Employee Attrition Predictor", layout="wide")

MODEL_PATH = Path("outputs/logistic_regression_attrition_model.joblib")
METRICS_PATH = Path("outputs/model_metrics.json")
DATA_PATH = Path("data/processed/attrition_clean.csv")
FIG_PATH = Path("outputs/figures/key_attrition_drivers.png")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


def prepare_features(df_input: pd.DataFrame, feature_names) -> pd.DataFrame:
    df_feat = df_input.copy()
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
    return X.reindex(columns=feature_names, fill_value=0)


model = load_model()
df = load_data()

st.title("Employee Attrition Risk Dashboard")
st.write("Predict individual employee flight risk, score batches, and review key attrition drivers.")

tab1, tab2, tab3 = st.tabs(["Single Employee Prediction", "Batch Risk Scoring", "EDA & Model Metrics"])

# --- TAB 1: Single Prediction ---
with tab1:
    st.subheader("Predict Attrition Risk for One Employee")
    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", 18, 65, 32)
        monthly_income = st.number_input("Monthly Income ($)", 1000, 25000, 5000, step=250)
        overtime = st.selectbox("OverTime", sorted(df["OverTime"].unique()))
        marital_status = st.selectbox("Marital Status", sorted(df["MaritalStatus"].unique()))

    with col2:
        department = st.selectbox("Department", sorted(df["Department"].unique()))
        job_role = st.selectbox("Job Role", sorted(df["JobRole"].unique()))
        business_travel = st.selectbox("Business Travel", sorted(df["BusinessTravel"].unique()))
        num_companies = st.number_input("Num Companies Worked", 0, 15, 2)

    with col3:
        total_working_years = st.number_input("Total Working Years", 0, 45, 8)
        years_at_company = st.number_input("Years At Company", 0, 40, 4)
        years_since_promo = st.number_input("Years Since Last Promotion", 0, 20, 1)
        threshold = st.slider("High-Risk Threshold", 0.10, 0.90, 0.40, 0.05)

    if st.button("Calculate Attrition Risk", type="primary"):
        # Start from dataset median/mode defaults so all columns exist
        base_row = df.drop(columns=["Attrition", "Attrition_Flag"], errors="ignore").mode().iloc[[0]].copy()
        for c in base_row.select_dtypes(include="number").columns:
            base_row[c] = df[c].median()

        base_row["Age"] = age
        base_row["MonthlyIncome"] = monthly_income
        base_row["OverTime"] = overtime
        base_row["MaritalStatus"] = marital_status
        base_row["Department"] = department
        base_row["JobRole"] = job_role
        base_row["BusinessTravel"] = business_travel
        base_row["NumCompaniesWorked"] = num_companies
        base_row["TotalWorkingYears"] = total_working_years
        base_row["YearsAtCompany"] = years_at_company
        base_row["YearsSinceLastPromotion"] = years_since_promo

        X_single = prepare_features(base_row, model.feature_names_in_)
        prob = float(model.predict_proba(X_single)[:, 1][0])

        st.metric("Predicted Attrition Risk", f"{prob * 100:.1f}%")
        if prob >= threshold:
            st.error(f"High Risk (above {threshold:.2f} threshold)")
        else:
            st.success(f"Low Risk (below {threshold:.2f} threshold)")

# --- TAB 2: Batch Scoring ---
with tab2:
    st.subheader("Score Employees in Bulk")
    batch_thresh = st.slider("Batch Risk Threshold", 0.10, 0.90, 0.40, 0.05, key="batch_thresh")
    uploaded_file = st.file_uploader("Upload a cleaned employee CSV (or leave empty to score default dataset)", type=["csv"])

    target_df = pd.read_csv(uploaded_file) if uploaded_file else df
    X_batch = prepare_features(target_df, model.feature_names_in_)
    probs = model.predict_proba(X_batch)[:, 1]

    scored = target_df[["OverTime", "JobRole", "MonthlyIncome"]].copy()
    if "Attrition" in target_df.columns:
        scored.insert(0, "Attrition", target_df["Attrition"])
    scored["Attrition_Risk_Prob"] = probs.round(3)
    scored["Predicted_High_Risk"] = (probs >= batch_thresh).astype(int)
    scored = scored.sort_values("Attrition_Risk_Prob", ascending=False)

    st.write(f"Flagged **{int(scored['Predicted_High_Risk'].sum())}** of **{len(scored)}** employees.")
    st.dataframe(scored.head(20), use_container_width=True)

    st.download_button(
        label="Download Full Scored CSV",
        data=scored.to_csv(index=False).encode("utf-8"),
        file_name="scored_employees.csv",
        mime="text/csv",
    )

# --- TAB 3: EDA & Model Metrics ---
with tab3:
    st.subheader("Model Evaluation Metrics")
    if METRICS_PATH.exists():
        with open(METRICS_PATH) as f:
            metrics = json.load(f)
        st.dataframe(pd.DataFrame(metrics).T, use_container_width=True)

    st.subheader("Key Attrition Drivers")
    if FIG_PATH.exists():
        st.image(str(FIG_PATH), use_container_width=True)