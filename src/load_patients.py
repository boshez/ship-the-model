import pandas as pd
from src.session import get_session

def main():
    session = get_session()
    df = pd.read_csv("data/synthetic_patients.csv")
    session.write_pandas(df, "PATIENTS", auto_create_table=True, overwrite=True)
    print(session.table("PATIENTS").count(), "rows loaded")

if __name__ == "__main__":
    main()
