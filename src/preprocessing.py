import pandas as pd

DROP_COLS = ["EmployeeCount", "Over18", "StandardHours", "EmployeeNumber"]


def load_and_clean_attrition(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df_clean = df.drop(columns=[c for c in DROP_COLS if c in df.columns]).copy()
    df_clean["Attrition_Flag"] = (df_clean["Attrition"] == "Yes").astype(int)
    return df_clean