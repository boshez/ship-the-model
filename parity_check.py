import joblib
from src.session import get_session

bundle = joblib.load("readmission_model.joblib")
model = bundle["model"]
feature_cols = bundle["feature_cols"]

session = get_session()

query = """
SELECT *,
    PREDICT_READMISSION("age", "frailty_score", "prior_admission_count",
                         "medication_count", "length_of_stay", "total_cost",
                         "HIGH_UTILIZER", "POLYPHARMACY", "LONG_STAY",
                         "IS_HOME", "IS_HOME_HEALTH", "IS_SNF", "IS_AMA") AS snowflake_score
FROM admissions.ml.patient_features
LIMIT 10
"""

df = session.sql(query).to_pandas()

X = df[feature_cols]
df["local_score"] = model.predict_proba(X)[:, 1]
df["diff"] = (df["SNOWFLAKE_SCORE"] - df["local_score"]).abs()

print(df[["patient_id", "SNOWFLAKE_SCORE", "local_score", "diff"]])
print("\nMax difference:", df["diff"].max())