"""Generate a small, recruiter-readable EDA report."""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from data import load_and_clean

def main(data_path, output_dir):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(data_path)
    summary = [
        "# EDA summary", "",
        f"- Rows: {len(df):,}", f"- Columns: {df.shape[1]}",
        f"- Churn rate: {df.Churn.mean():.2%}",
        f"- Missing TotalCharges after numeric conversion: {df.TotalCharges.isna().sum()}",
        "", "These are dataset facts, not model results.",
    ]
    (out / "eda_summary.md").write_text("\n".join(summary))
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    sns.countplot(data=df, x="Churn", ax=axes[0], color="#73a89a")
    axes[0].set_xticks([0, 1], ["Stayed", "Churned"]); axes[0].set_title("Class balance")
    sns.histplot(data=df, x="tenure", hue="Churn", bins=24, ax=axes[1], element="step")
    axes[1].set_title("Tenure by churn")
    rate = df.groupby("Contract", observed=True).Churn.mean().sort_values(ascending=False)
    sns.barplot(x=rate.values, y=rate.index, ax=axes[2], color="#73a89a")
    axes[2].set_xlabel("Churn rate"); axes[2].set_ylabel(""); axes[2].set_title("Churn by contract")
    fig.tight_layout(); fig.savefig(out / "eda_overview.png", dpi=170); plt.close(fig)
if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--data", default="data/raw/Telco-Customer-Churn.csv"); p.add_argument("--output", default="reports")
    a=p.parse_args(); main(a.data, a.output)
