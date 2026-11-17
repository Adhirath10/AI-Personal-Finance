# Explainability (SHAP)

This module explains the existing saved models. It never trains, retrains, or
overwrites them. SQLite connections are read-only; explanation artifacts are
written separately to `data/processed/explainability/` by default.
SHAP is pinned to 0.47.2 in the root and AI-service requirements.

Run commands from the project root:

```powershell
python ai-service\explainability\run.py forecast-global
python ai-service\explainability\run.py forecast-user --user-id U001
python ai-service\explainability\run.py categorize --description "Cafe purchase" --merchant "Cafe"
python ai-service\explainability\run.py categorize-global
python ai-service\explainability\run.py anomaly-transaction --transaction-id T0000014
python ai-service\explainability\run.py anomaly-global --scope personal --user-id U001
```

Use `--database-path`, `--model-dir`, and `--output-dir` to select an existing
database, the corresponding saved model directory, or a separate output
location. The default database path and model directories match the existing
AI service. `forecast-user` explains the input row selected by the current
standalone forecasting predictor; it intentionally does not change that
predictor's behavior.

## Explainability methods

- **Expense forecasting:** `TreeExplainer` explains the saved
  `RandomForestRegressor` in expense-output units, using tree-path-dependent
  reference values. Global output includes mean absolute SHAP values and
  bar/beeswarm plots. A per-user explanation includes the forecast month,
  feature values, signed contributions, base value, and a waterfall plot.
- **Transaction categorization:** `LinearExplainer` uses the saved TF-IDF
  vectorizer and Logistic Regression classifier. It explains a selected class's
  decision score in TF-IDF unigram/bigram space, not probability changes.
  The current database is split again with the original fixed random seed to
  reconstruct training-only background rows and held-out examples. The module
  validates the classifier classes and vocabulary before using that reference.
  The model artifacts do not contain a training-data fingerprint or original
  row order, so this reconstruction cannot prove it is the exact historical
  split; the JSON output marks this limitation. If the classifier's classes or
  vocabulary do not match, the module fails instead of using that reference.
- **Behavioral anomaly detection:** permutation SHAP wraps the saved
  behavioral Isolation Forest pipeline's negated `decision_function`. SHAP
  values therefore describe contributions to its continuous anomaly score
  (higher means more unusual), not a fraud probability. Categorical and
  behavioral fields are passed through the existing fitted preprocessor and
  `prepare_experiment_features` builder. Personal, transaction-type, and pooled
  model summaries are kept separate. The training-period background is
  reconstructed without labels and checked against the saved training-row
  count; a count mismatch causes an explicit error. The model artifact does not
  save a feature-data fingerprint, so an unchanged count cannot prove that the
  reconstructed records exactly match the historic training snapshot.

Every method reuses existing feature transformations and reports its feature
names. Targets and anomaly labels are not SHAP inputs. Behavioral transaction
features are computed from prior transactions only; an individual explanation
uses no rows later than the transaction being explained. Global model-agnostic
anomaly explanations are approximate and more expensive than the tree and
linear methods.

## Limitations

SHAP describes how a fitted model's output changes relative to a chosen
reference distribution. It does not establish cause, correctness, fairness,
fraud, or financial suitability. Correlated lag/rolling features and correlated
TF-IDF n-grams can share or shift attribution. Anomaly explanations describe
the Isolation Forest score and its reference cohort; contamination is not a
probability. The rule-based recommendation engine should be explained through
its triggered rules and supporting values, not SHAP.

The existing standalone forecasting predictor and the recommendation engine
construct forecasts through different inference paths. This module preserves
the standalone predictor's current behavior and does not silently alter either
path. See the project-level integration before interpreting forecasts from
another serving path.

## Outputs

Global runs create JSON metadata, a CSV of mean absolute SHAP values, and
visualizations. Individual runs create JSON feature values and attributions
plus a waterfall plot. The categorization output omits the original input text
from saved files. These outputs can still contain sensitive financial details;
choose the output directory and access controls accordingly.
