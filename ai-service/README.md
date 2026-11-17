# AI Service - Transaction Categorization

This module builds the initial machine-learning baseline for transaction categorization using the existing SQLite dataset in the project.

## Scope

- Reads transaction data from the existing SQLite database
- Cleans and combines description + merchant text
- Uses TF-IDF + Logistic Regression
- Trains and evaluates a multiclass category classifier
- Saves model artifacts and evaluation outputs

This module does not create synthetic data and does not modify the original database.
