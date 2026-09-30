"""Feature engineering that never leaves Snowflake.

Every transformation below is lazily built into a SQL query; nothing
executes until an action (.count(), .show(), save) is called. Print the
generated SQL with df.queries -- that's the pushdown proof.
"""
import snowflake.snowpark.functions as F
from snowflake.snowpark import Session

def build_features(session: Session):
    p = session.table("PATIENTS")

    feats = (
        p.with_column("HIGH_UTILIZER",
            F.when(F.col('"prior_admission_count"') >= 2, 1).otherwise(0))
         .with_column("POLYPHARMACY",
            F.when(F.col('"medication_count"') >= 10, 1).otherwise(0))
         .with_column("LONG_STAY",
            F.when(F.col('"length_of_stay"') >= 7, 1).otherwise(0))
         .with_column("IS_HOME",
            F.when(F.col('"discharge_destination"') == "Home", 1).otherwise(0))
         .with_column("IS_HOME_HEALTH",
            F.when(F.col('"discharge_destination"') == "Home Health Care", 1).otherwise(0))
         .with_column("IS_SNF",
            F.when(F.col('"discharge_destination"') == "Skilled Nursing Facility", 1).otherwise(0))
         .with_column("IS_AMA",
            F.when(F.col('"discharge_destination"') == "Against Medical Advice", 1).otherwise(0))
    )

    keep = ['"patient_id"', '"age"', '"frailty_score"', '"prior_admission_count"',
            '"medication_count"', '"length_of_stay"', '"total_cost"',
            "HIGH_UTILIZER", "POLYPHARMACY", "LONG_STAY",
            "IS_HOME", "IS_HOME_HEALTH", "IS_SNF", "IS_AMA",
            '"is_30day_readmit"']
    feats = feats.select(*keep)
    feats.write.save_as_table("PATIENT_FEATURES", mode="overwrite")
    return feats

if __name__ == "__main__":
    from src.session import get_session
    s = get_session()
    f = build_features(s)
    print("rows:", f.count())
    print("--- generated SQL ---")
    print(f.queries["queries"][-1])
    f.group_by('"is_30day_readmit"').count().show()
