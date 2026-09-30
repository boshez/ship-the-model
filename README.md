# Project 9: Snowpark UDF Model Scoring

A 30-day hospital readmission risk classifier, trained locally and deployed as a
native Snowflake Python UDF (`PREDICT_READMISSION`) — scoring happens entirely
in SQL, with no external model-serving infrastructure required.

## Pipeline

Snowflake table (PATIENTS)
→ Snowpark feature engineering (pushdown SQL, nothing leaves the warehouse)
→ CalibratedClassifierCV (isotonic calibration), trained locally
→ registered as a permanent Python UDF via session.udf.register(...)
→ scored entirely in SQL: SELECT PREDICT_READMISSION(...) FROM patient_features


## Model performance

- **AUC: 0.572** — a weak but real signal, not a strong classifier. This is
  reported honestly rather than oversold; the point of this project is the
  deployment pattern (train locally, serve inside the warehouse), not
  state-of-the-art predictive performance.
- Calibrated with `CalibratedClassifierCV(method="isotonic")`.

## Risk tiers

Patients are split by **percentile rank**, not a fixed score threshold —
because in a real hospital setting, care-management capacity is the actual
constraint, not an arbitrary probability cutoff.

| Tier   | Patients | % of total | Score range     |
|--------|---------:|-----------:|------------------|
| High   |      954 |       9.8% | 0.479 – 0.611    |
| Medium |    1,956 |      20.2% | 0.277 – 0.479    |
| Low    |    6,789 |      70.0% | 0.134 – 0.276    |

## Known quirk: score plateau at the top

Several of the highest-risk patients share the exact same score (0.6109…).
This is expected behavior from isotonic calibration — it maps a range of raw
model outputs onto flat plateaus — combined with the model's weak
discrimination (AUC 0.572). Not a bug.

## Parity check: local model vs. deployed UDF

A parity check (comparing the UDF's live SQL output against the same model's
`predict_proba` run locally) surfaced a genuine deployment finding: predicted
probabilities differed by up to **0.13** between the two, despite both sides
running identical scikit-learn versions (1.9.1).

Root cause: **Snowflake Python UDFs snapshot their imported files at creation
time.** The model file in the stage was last modified *after* the UDF was
registered, so the live UDF has been serving an older model snapshot than
what's currently sitting in the stage — overwriting a staged file does not
automatically refresh an already-created UDF.

This is a real production consideration for anyone deploying ML via Snowpark
UDFs: a model update requires an explicit `CREATE OR REPLACE FUNCTION` to take
effect, not just a file overwrite in the stage.

## Stack

Snowflake · Snowpark Python · scikit-learn · Python UDFs