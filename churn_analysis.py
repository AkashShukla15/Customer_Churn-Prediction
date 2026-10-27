"""
Customer Churn Prediction  |  SQL + Python + Machine Learning + Power BI
=========================================================================
Dataset : Telco Customer Churn (Kaggle) -> save as  data/telco_churn.csv
          https://www.kaggle.com/datasets/blastchar/telco-customer-churn
          (If the file is missing, a synthetic Telco-like dataset is generated
           so the script still runs end to end.)

Run     : python churn_analysis.py
Outputs : outputs/  (charts, model comparison, Power BI CSV, saved model)
"""

import os
import sqlite3
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = "data/telco_churn.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)


# --------------------------------------------------------------------------
# 1. LOAD DATA
# --------------------------------------------------------------------------
def make_synthetic_telco(n=7043, seed=RANDOM_STATE):
    """Fallback dataset with the same columns as the Kaggle Telco file."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": rng.choice(["Yes", "No"], n),
        "Dependents": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
        "tenure": rng.integers(0, 73, n),
        "PhoneService": rng.choice(["Yes", "No"], n, p=[0.9, 0.1]),
        "InternetService": rng.choice(["DSL", "Fiber optic", "No"], n,
                                      p=[0.34, 0.44, 0.22]),
        "Contract": rng.choice(["Month-to-month", "One year", "Two year"], n,
                               p=[0.55, 0.21, 0.24]),
        "PaperlessBilling": rng.choice(["Yes", "No"], n),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"], n),
    })
    df["MultipleLines"] = np.where(df.PhoneService == "No", "No phone service",
                                   rng.choice(["Yes", "No"], n))
    for col in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        df[col] = np.where(df.InternetService == "No", "No internet service",
                           rng.choice(["Yes", "No"], n))
    base = 20 + 25 * (df.InternetService == "DSL") + \
        55 * (df.InternetService == "Fiber optic")
    df["MonthlyCharges"] = (base + rng.normal(0, 6, n)).round(2).clip(18, 120)
    df["TotalCharges"] = (df.MonthlyCharges * df.tenure).round(2).astype(str)
    df.loc[df.tenure == 0, "TotalCharges"] = " "          # same quirk as real data
    logit = (-1.0 + 1.6 * (df.Contract == "Month-to-month")
             - 1.2 * (df.Contract == "Two year")
             + 0.9 * (df.InternetService == "Fiber optic")
             + 0.7 * (df.PaymentMethod == "Electronic check")
             - 0.04 * df.tenure + 0.012 * (df.MonthlyCharges - 65)
             + 0.4 * (df.TechSupport == "No"))
    df["Churn"] = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return df


if os.path.exists(DATA_PATH):
    raw = pd.read_csv(DATA_PATH)
    print(f"Loaded real dataset: {raw.shape}")
else:
    raw = make_synthetic_telco()
    raw.to_csv(DATA_PATH, index=False)
    print(f"[!] {DATA_PATH} not found -> generated synthetic data {raw.shape}. "
          "Replace it with the Kaggle file for real results.")

# --------------------------------------------------------------------------
# 2. DATA CLEANING
# --------------------------------------------------------------------------
df = raw.copy()
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
# Blank TotalCharges = customers with tenure 0 (never billed) -> 0
df["TotalCharges"] = df["TotalCharges"].fillna(0)
df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
df["Churn_Flag"] = (df["Churn"] == "Yes").astype(int)
df = df.drop_duplicates(subset="customerID")

print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])
print(f"Overall churn rate: {df.Churn_Flag.mean():.1%}")

# Feature engineering
df["TenureGroup"] = pd.cut(df["tenure"], [-1, 12, 24, 48, 72],
                           labels=["0-12m", "13-24m", "25-48m", "49-72m"])
service_cols = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]
df["NumServices"] = (df[service_cols] == "Yes").sum(axis=1)
df["AvgMonthlySpend"] = df["TotalCharges"] / df["tenure"].replace(0, 1)

# --------------------------------------------------------------------------
# 3. SQL ANALYSIS (SQLite, same SQL works in MySQL / PostgreSQL)
# --------------------------------------------------------------------------
conn = sqlite3.connect(os.path.join(OUT_DIR, "churn.db"))
df.assign(TenureGroup=df["TenureGroup"].astype(str)).to_sql(
    "customers", conn, if_exists="replace", index=False)

queries = {
    "Churn rate by contract type": """
        SELECT Contract,
               COUNT(*)                                   AS customers,
               SUM(Churn_Flag)                            AS churned,
               ROUND(100.0 * SUM(Churn_Flag) / COUNT(*), 1) AS churn_pct
        FROM customers GROUP BY Contract ORDER BY churn_pct DESC;""",
    "Churn rate by internet service": """
        SELECT InternetService, COUNT(*) AS customers,
               ROUND(100.0 * AVG(Churn_Flag), 1) AS churn_pct,
               ROUND(AVG(MonthlyCharges), 2)     AS avg_monthly
        FROM customers GROUP BY InternetService ORDER BY churn_pct DESC;""",
    "Churn rate by tenure group": """
        SELECT TenureGroup, COUNT(*) AS customers,
               ROUND(100.0 * AVG(Churn_Flag), 1) AS churn_pct
        FROM customers GROUP BY TenureGroup ORDER BY MIN(tenure);""",
    "Revenue lost to churn (monthly)": """
        SELECT ROUND(SUM(CASE WHEN Churn_Flag = 1 THEN MonthlyCharges END), 0) AS lost_monthly_revenue,
               ROUND(SUM(MonthlyCharges), 0)                                    AS total_monthly_revenue,
               ROUND(100.0 * SUM(CASE WHEN Churn_Flag = 1 THEN MonthlyCharges END)
                     / SUM(MonthlyCharges), 1)                                  AS pct_revenue_lost
        FROM customers;""",
    "Churn by payment method (window rank)": """
        SELECT PaymentMethod,
               ROUND(100.0 * AVG(Churn_Flag), 1) AS churn_pct,
               RANK() OVER (ORDER BY AVG(Churn_Flag) DESC) AS risk_rank
        FROM customers GROUP BY PaymentMethod;""",
    "High-risk segment: month-to-month + fiber + no tech support": """
        SELECT COUNT(*) AS customers,
               ROUND(100.0 * AVG(Churn_Flag), 1) AS churn_pct
        FROM customers
        WHERE Contract = 'Month-to-month'
          AND InternetService = 'Fiber optic'
          AND TechSupport = 'No';""",
}
print("\n================ SQL INSIGHTS ================")
for title, q in queries.items():
    print(f"\n--- {title}")
    print(pd.read_sql(q, conn).to_string(index=False))
conn.close()

# --------------------------------------------------------------------------
# 4. EDA CHARTS
# --------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
for ax, col in zip(axes.flat, ["Contract", "InternetService", "PaymentMethod",
                               "TechSupport", "TenureGroup", "SeniorCitizen"]):
    rate = df.groupby(col, observed=True)["Churn_Flag"].mean().mul(100)
    rate.plot.bar(ax=ax, color="#d9534f")
    ax.set_title(f"Churn % by {col}")
    ax.set_ylabel("Churn %")
    ax.tick_params(axis="x", rotation=25)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_churn_rates.png", dpi=130)
plt.close()

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
for flag, color, label in [(0, "#5cb85c", "Stayed"), (1, "#d9534f", "Churned")]:
    ax[0].hist(df.loc[df.Churn_Flag == flag, "tenure"], bins=30, alpha=.6,
               color=color, label=label)
    ax[1].hist(df.loc[df.Churn_Flag == flag, "MonthlyCharges"], bins=30,
               alpha=.6, color=color, label=label)
ax[0].set_title("Tenure distribution"); ax[1].set_title("Monthly charges")
ax[0].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/eda_distributions.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 5. PREPROCESSING + MODEL COMPARISON
# --------------------------------------------------------------------------
y = df["Churn_Flag"]
X = df.drop(columns=["customerID", "Churn", "Churn_Flag"])
X["TenureGroup"] = X["TenureGroup"].astype(str)

num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000,
                                              class_weight="balanced"),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8,
                                            class_weight="balanced",
                                            random_state=RANDOM_STATE, n_jobs=-1),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}
# class_weight="balanced" handles the ~27% churn imbalance without SMOTE.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
rows, fitted = [], {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", clf)])
    cv_proba = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                 method="predict_proba")[:, 1]
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    rows.append({"Model": name,
                 "CV ROC-AUC": roc_auc_score(y_train, cv_proba),
                 "Test ROC-AUC": roc_auc_score(y_test, proba),
                 "Precision": precision_score(y_test, pred),
                 "Recall": recall_score(y_test, pred),
                 "F1": f1_score(y_test, pred)})
    fitted[name] = (pipe, cv_proba)

results = pd.DataFrame(rows).sort_values("Test ROC-AUC", ascending=False)
print("\n================ MODEL COMPARISON (threshold 0.5) ================")
print(results.round(3).to_string(index=False))
results.round(4).to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

best_name = results.iloc[0]["Model"]
best_pipe, best_cv_proba = fitted[best_name]
print(f"\nBest model by ROC-AUC: {best_name}")

# --------------------------------------------------------------------------
# 6. THRESHOLD TUNING (business logic: missing a churner costs more
#    than contacting a loyal customer, so we favour recall)
# --------------------------------------------------------------------------
thresholds = np.arange(0.2, 0.71, 0.01)
f2 = []
for t in thresholds:   # F2 weights recall 2x more than precision
    p = precision_score(y_train, best_cv_proba >= t, zero_division=0)
    r = recall_score(y_train, best_cv_proba >= t)
    f2.append(5 * p * r / (4 * p + r) if (4 * p + r) else 0)
best_t = float(thresholds[int(np.argmax(f2))])   # chosen on TRAIN CV only
print(f"Chosen decision threshold (max F2 on CV): {best_t:.2f}")

proba_test = best_pipe.predict_proba(X_test)[:, 1]
pred_test = (proba_test >= best_t).astype(int)
print("\n================ FINAL TEST REPORT ================")
print(classification_report(y_test, pred_test, target_names=["Stayed", "Churned"]))
print(f"ROC-AUC: {roc_auc_score(y_test, proba_test):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ConfusionMatrixDisplay.from_predictions(y_test, pred_test, ax=ax[0],
                                        display_labels=["Stayed", "Churned"],
                                        cmap="Reds")
ax[0].set_title(f"Confusion matrix (threshold {best_t:.2f})")
for name, (pipe, _) in fitted.items():
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=ax[1], name=name)
ax[1].plot([0, 1], [0, 1], "k--", alpha=.4)
ax[1].set_title("ROC curves")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/model_evaluation.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE (permutation, on raw columns -> easy to explain)
# --------------------------------------------------------------------------
perm = permutation_importance(best_pipe, X_test, y_test, scoring="roc_auc",
                              n_repeats=8, random_state=RANDOM_STATE, n_jobs=-1)
imp = (pd.Series(perm.importances_mean, index=X_test.columns)
       .sort_values(ascending=False))
print("\nTop churn drivers (drop in ROC-AUC when shuffled):")
print(imp.head(8).round(4).to_string())
imp.head(10).sort_values().plot.barh(color="#337ab7", figsize=(8, 5))
plt.title("Top 10 churn drivers"); plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/feature_importance.png", dpi=130)
plt.close()

# --------------------------------------------------------------------------
# 8. SCORE ALL CUSTOMERS + EXPORT FOR POWER BI
# --------------------------------------------------------------------------
best_pipe.fit(X, y)                      # refit on all data for production scoring
df["ChurnProbability"] = best_pipe.predict_proba(X)[:, 1].round(4)
df["RiskSegment"] = pd.cut(df["ChurnProbability"], [-0.01, 0.3, 0.6, 1.0],
                           labels=["Low", "Medium", "High"])
df["TenureGroup"] = df["TenureGroup"].astype(str)
df.to_csv(f"{OUT_DIR}/powerbi_churn_scored.csv", index=False)
joblib.dump({"pipeline": best_pipe, "threshold": best_t},
            f"{OUT_DIR}/churn_model.joblib")

high = df[df.RiskSegment == "High"]
print(f"\nHigh-risk customers: {len(high)} "
      f"(monthly revenue at risk: {high.MonthlyCharges.sum():,.0f})")
print(f"\nDone. Files saved in ./{OUT_DIR}/")
