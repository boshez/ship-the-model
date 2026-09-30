"""Register the model as a permanent Snowflake UDF."""
from snowflake.snowpark.types import FloatType, IntegerType
from src.session import get_session

def main():
    session = get_session()

    def predict_readmission(age: int, frailty_score: int, prior_admission_count: int,
                             medication_count: int, length_of_stay: int, total_cost: float,
                             high_utilizer: int, polypharmacy: int, long_stay: int,
                             is_home: int, is_home_health: int, is_snf: int, is_ama: int) -> float:
        import sys, os, joblib
        if not hasattr(predict_readmission, "_bundle"):
            d = sys._xoptions["snowflake_import_directory"]
            predict_readmission._bundle = joblib.load(
                os.path.join(d, "readmission_model.joblib"))
        b = predict_readmission._bundle
        import pandas as pd
        row = pd.DataFrame([[age, frailty_score, prior_admission_count, medication_count,
                              length_of_stay, total_cost, high_utilizer, polypharmacy,
                              long_stay, is_home, is_home_health, is_snf, is_ama]],
                            columns=b["feature_cols"])
        return float(b["model"].predict_proba(row)[0, 1])

    session.udf.register(
        func=predict_readmission,
        name="PREDICT_READMISSION",
        return_type=FloatType(),
        input_types=[IntegerType(), IntegerType(), IntegerType(), IntegerType(),
                     IntegerType(), FloatType(), IntegerType(), IntegerType(),
                     IntegerType(), IntegerType(), IntegerType(), IntegerType(), IntegerType()],
        packages=["scikit-learn==1.9.1", "pandas", "joblib"],
        imports=["@ml_stage/readmission_model.joblib"],
        is_permanent=True,
        stage_location="@ml_stage",
        replace=True,
        runtime_version="3.11",
    )
    print("UDF registered.")

if __name__ == "__main__":
    main()
