import os
from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()

def get_session() -> Session:
    return Session.builder.configs({
        "account":   os.environ["SNOWFLAKE_ACCOUNT"],
        "user":      os.environ["SNOWFLAKE_USER"],
        "password":  os.environ["SNOWFLAKE_PASSWORD"],
        "warehouse": "PIPELINE_WH",
        "database":  "ADMISSIONS",
        "schema":    "ML",
    }).create()
