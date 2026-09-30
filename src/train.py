"""Train the readmission model on features pulled from Snowflake."""
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV
from src.session import get_session

FEATURE_COLS = ["age", "frailty_score", "prior_admission_count",
                "medication_count", "length_of_stay", "total_cost",
                "HIGH_UTILIZER", "POLYPHARMACY", "LONG_STAY",
                "IS_HOME", "IS_HOME_HEALTH", "IS_SNF", "IS_AMA"]
TARGET_COL = "is_30day_readmit"

def main():
    session = get_session()
    df = session.table("PATIENT_FEATURES").to_pandas()
    X, y = df[FEATURE_COLS], df[TARGET_COL]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25,
                                                stratify=y, random_state=42)
    base = HistGradientBoostingClassifier(random_state=42)
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"AUC: {roc_auc_score(y_te, proba):.3f}")
    print(f"Brier: {brier_score_loss(y_te, proba):.3f}")

    joblib.dump({"model": model, "feature_cols": FEATURE_COLS},
                "readmission_model.joblib")

    session.file.put("readmission_model.joblib", "@ml_stage",
                      auto_compress=False, overwrite=True)
    print(session.sql("LIST @ml_stage").collect())

if __name__ == "__main__":
    main()
