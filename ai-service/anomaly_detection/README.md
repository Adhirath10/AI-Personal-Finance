# Transaction Anomaly Detection

This module detects transactions that are unusual relative to a user's own
spending history using scikit-learn's Isolation Forest. It reads the existing
SQLite database in read-only mode and does not update transaction records.

## Database and features

By default, the module reads
`data/database/personal_finance.db`. Set `FINANCE_DB_PATH` or pass
`--database-path` to use another existing database.

The model expects the existing `transactions` columns:

- `transaction_id`, `user_id`, and `transaction_date` identify, group, and
  temporally split rows; identifiers and dates are not model inputs.
- `amount` is transformed with signed `log1p` to reduce right skew.
- `category`, `subcategory`, `payment_mode`, and `account_type` are one-hot
  encoded.
- `is_recurring` and `balance_after_transaction` are numeric inputs.
- Day of week and month are derived from `transaction_date` and encoded as
  cyclic sine/cosine values.

Numeric values are median-imputed and standardized. Categorical nulls are
represented by an explicit missing-value category, and unseen categories at
prediction time are safely ignored by the fitted encoder. All preprocessing
is part of the persisted scikit-learn pipelines.

Training uses expense transactions only. `transaction_type` selects those
rows and is therefore not a feature (it would be constant in the training
data). There is a pooled expense fallback and a separate model for each user
with at least 50 expense transactions. Thus `user_id` is not encoded as a
model feature, while a user's baseline is learned from their expense history
where sufficient data exists. The prediction helper returns only expenses
when loading this expense-only model.

`is_anomaly` and `anomaly_type` are not loaded during training or prediction.
Evaluation loads them separately and uses them only to calculate reference
metrics. The existing labels have not been independently verified as
real-world anomalies.

## Install

From the project root:

```powershell
python -m pip install -r ai-service\anomaly_detection\requirements.txt
```

## Train

```powershell
python ai-service\anomaly_detection\train.py
```

Training is deterministic (`random_state=42`) and defaults to an Isolation
Forest contamination of `0.02`. Adjust contamination for the desired alert
rate, for example:

```powershell
python ai-service\anomaly_detection\train.py --contamination 0.03
```

The model bundle and metadata are saved to:

- `ai-service/models/anomaly_detection/isolation_forest_models.joblib`
- `ai-service/models/anomaly_detection/model_metadata.json`

The bundle includes each fitted preprocessor and Isolation Forest. Retrain
after changing the database or contamination setting.

## Predict

```powershell
python ai-service\anomaly_detection\predict.py --user-id U001
python ai-service\anomaly_detection\predict.py --transaction-id T0000014
```

Choose exactly one selector. `--transaction-id` prints one expense
transaction's result. `--user-id` prints results for all of that user's
expense transactions. Each result includes the transaction ID, user ID,
amount, category, normal/anomaly prediction, and score. The anomaly score is
the negated Isolation Forest decision function; a larger score indicates a
more unusual transaction. Income transactions are not supported by this
expense-only model. The SQLite database is never modified.

## Evaluate

```powershell
python ai-service\anomaly_detection\evaluate.py
```

Evaluation holds out the latest 20% of unique transaction dates. Models and
preprocessing are fit only on earlier expense transactions, then compared
with `is_anomaly` labels in the held-out window. The evaluation generates
accuracy, precision, recall, F1, a classification report, a confusion matrix,
a confusion-matrix PNG, and a text report. By default the reports are saved
to `ai-service/models/anomaly_detection/evaluation_report.txt` and
`ai-service/models/anomaly_detection/confusion_matrix.png`. Use
`--test-fraction`, `--output`, or `--confusion-matrix` to customize them.
The stored labels have not been independently verified as ground truth, so
reported metrics describe agreement with those annotations, not validated
anomaly detection or fraud detection performance.

## Behavioral feature experiment

Run the isolated experiment from the project root:

```powershell
python ai-service\anomaly_detection\behavioral_experiment.py
```

It compares the existing baseline pipeline with an experimental pipeline using
prior-only per-user behavioral features. Historical averages, sample standard
deviations, category statistics, merchant familiarity, and transaction
frequencies are calculated from transactions strictly earlier than each
transaction. Rows for the same user and timestamp share the same prior state.
The existing chronological latest-20%-of-dates holdout is preserved; features
for test rows can use earlier unlabeled transaction attributes, but never
future rows or stored labels. Preprocessing is fit on training rows only.

The experiment saves its model, metadata, and comparison report under
`ai-service\models\anomaly_detection_behavioral_experiment\`, separate from
the current model directory. It does not modify the SQLite database or replace
the existing model. The report treats the existing labels as unverified
reference annotations, not verified fraud labels.

## Requirements and limitations

Isolation Forest is unsupervised; contamination influences its decision
threshold, not a learned fraud probability. Per-user models need sufficient
history, and users with fewer than 50 expense rows use the pooled fallback.
Evaluate alert volume and quality with domain review before relying on alerts.
