from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

Path("outputs/figures").mkdir(parents=True, exist_ok=True)
df = pd.read_csv("data/processed/attrition_clean.csv")

sns.set_theme(style="whitegrid")

drivers = ["OverTime", "Department", "JobRole", "BusinessTravel", "MaritalStatus"]
for col in drivers:
    summary = (
        df.groupby(col)["Attrition_Flag"]
        .agg(Employees="count", Attrition_Rate="mean")
        .sort_values("Attrition_Rate", ascending=False)
    )
    summary["Attrition_Rate"] = (summary["Attrition_Rate"] * 100).round(1)
    print(f"\n--- {col} ---")
    print(summary)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

sns.barplot(data=df, x="OverTime", y="Attrition_Flag", ax=axes[0], palette="Set2")
axes[0].set_title("Attrition Rate by Overtime")
axes[0].set_ylabel("Attrition Rate")

sns.boxplot(data=df, x="Attrition", y="MonthlyIncome", ax=axes[1], palette="Set2")
axes[1].set_title("Monthly Income Distribution by Attrition")

plt.tight_layout()
plt.savefig("outputs/figures/key_attrition_drivers.png", dpi=150)
plt.show()