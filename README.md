# AI-Powered Personal Finance Monitoring and Predictive Financial Management System

## Project Overview

This is a comprehensive final-year academic project that builds an AI-powered personal finance system capable of:

1. **Automatic Transaction Categorization** - Uses NLP and ML to categorize transactions
2. **Spending Behavior Analysis** - Identifies patterns, trends, and recurring expenses
3. **Expense Prediction** - Forecasts future spending using time-series models
4. **Anomaly Detection** - Identifies unusual transactions and spending patterns
5. **Budget Analysis** - Compares actual and predicted spending against budgets
6. **AI Recommendation Engine** - Generates personalized, explainable financial recommendations
7. **What-If Financial Simulator** - Allows scenario modeling and financial projections
8. **Explainable AI (XAI)** - Uses SHAP to explain saved forecasting, categorization, and behavioral anomaly models

## Project Structure

```
AI-Powered-Personal-Finance/
├── backend/                          # FastAPI backend service
│   ├── app/
│   │   ├── api/                      # API route endpoints
│   │   ├── database/                 # Database models and connections
│   │   ├── models/                   # SQLAlchemy ORM models
│   │   ├── schemas/                  # Pydantic request/response schemas
│   │   ├── services/                 # Business logic services
│   │   ├── ml/                       # ML/AI modules
│   │   │   ├── preprocessing/        # Data preprocessing
│   │   │   ├── categorization/       # Transaction categorization
│   │   │   ├── forecasting/          # Expense prediction
│   │   │   ├── anomaly_detection/    # Anomaly detection
│   │   │   └── recommendations/      # Recommendation engine
│   │   ├── analytics/                # Financial analytics
│   │   ├── simulator/                # What-if simulator
│   │   ├── config/                   # Configuration
│   │   └── main.py                   # FastAPI app entry point
│   └── requirements.txt              # Python dependencies
│
├── frontend/                         # React + Vite FinAI dashboard
│
├── data/
│   ├── raw/                          # Raw input data
│   ├── processed/                    # Processed data
│   └── database/                     # SQLite database directory
│
├── models/                           # Trained ML models
│   ├── categorization/               # Categorization models
│   ├── forecasting/                  # Forecasting models
│   ├── anomaly_detection/            # Anomaly detection models
│   └── recommendations/              # Recommendation models
│
├── notebooks/                        # Jupyter notebooks for analysis
│
├── tests/                            # Unit and integration tests
│
├── docs/                             # Documentation
│
├── .gitignore                        # Git ignore file
├── .env.example                      # Environment variables template
└── README.md                         # This file
```

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip or conda
- SQLite3
- Virtual environment tool (venv, conda)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI-Powered-Personal-Finance
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Configure environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your settings
   ```

### Running the Application

**Start the AI model service** (in a separate terminal from the project root):
```bash
python -m uvicorn api:app --app-dir ai-service --host 127.0.0.1 --port 8001
```

**Start the FastAPI backend** (in another terminal, with `.env` configured):
```bash
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

**Start the frontend dashboard** (in another terminal):
```powershell
npm --prefix frontend install
Copy-Item frontend\.env.example frontend\.env
npm --prefix frontend run dev
```

### Retrain the expense forecast

The forecaster predicts next month relative to each user's own 3-month average
(blended 50/50 with that average and kept within 0.6x-1.6x of it), so it works
for low and high spenders alike. Retrain it on your database once:

```bash
python ai-service/forecasting/train.py
```

It prints the comparison with the moving-average baselines and the previous
model, and writes `forecast_metadata.json` and `evaluation_report.txt` next to
the model. Restart the AI service afterwards.

### Retrain the transaction categorizer

The categorizer reads whole words and character pieces of words, so spelling
mistakes ("swigy", "zomto") and everyday merchants missing from the dataset
("Dominos", "Jio recharge", "Zerodha") still get a sensible category. Retrain it
once on your database:

```bash
python ai-service/categorization/train.py
```

The report (`ai-service/models/categorization/evaluation_report.txt`) compares
the previous word-only model with the new one on the normal test set, on the
same test set with spelling mistakes, and on merchants held out of training.
Guesses below 30% confidence are marked "please check" in the app. In
**Add transaction**, **Suggest** fills in the category from the model.

### Measure the anomaly detector as the app uses it

The app combines the Isolation Forest with checks against your own history
(large for the category, duplicate charge, share of income, new merchant), and
shows a model-only flag only when the score is unusual for that user too. To
compare the model alone, the checks alone and the combination against the
stored labels:

```bash
python ai-service/anomaly_detection/evaluate_hybrid.py
```

### Recent data for a demo

The dataset ends in 2025, so forecasts would be for months that have already
passed. To show current data for one user, generate recent months from that
user's own history: their merchants and categories, a monthly salary, and
spending that leaves a realistic 22–38% of income saved (run from the project root):

```bash
python scripts/add_recent_data.py
```

It uses the same database and user as the app (`DATABASE_PATH` and
`FINANCE_USER_ID` from `.env`); `--status` shows which ones and the user's latest
month. This adds complete months up to the end of last month, so the forecast is for the
current month. The script backs up the database to `data/database/backups/`
first. Generated rows have IDs starting with `DEMO-`. Re-running it replaces
them, and `--remove` deletes them:

```bash
python scripts/add_recent_data.py --remove
```

Restart the backend and AI service afterwards, then use **Run AI analysis** on
the overview to fill in every AI panel at once.

Set `FINANCE_USER_ID` in the root `.env` to an existing `users.user_id` in the
database, and keep both Python services bound to loopback. The frontend does
not use demo fallbacks; failed requests are shown as errors. The service
configuration, endpoint contract, request examples, and limitations are in
[docs/API_DOCUMENTATION.md](./docs/API_DOCUMENTATION.md). FastAPI's interactive
documentation is available at `http://127.0.0.1:8000/docs`.

## System Architecture

### Data Flow

```
User Financial Data
    ↓
Data Collection (Upload/Manual Entry)
    ↓
Data Cleaning & Preprocessing
    ↓
Transaction Categorization (ML)
    ↓
Feature Engineering
    ↓
Spending Analysis
    ↓
Recurring Expense Detection
    ↓
Expense Prediction (ML)
    ↓
Anomaly Detection (ML)
    ↓
Budget Analysis & Comparison
    ↓
Recommendation Generation (ML)
    ↓
What-If Simulation
    ↓
Interactive Dashboard
    ↓
User Feedback
```

## Core Components

### 1. Transaction Categorization
- **Input**: Transaction description, merchant name, amount
- **Output**: Predicted category with confidence score
- **Technology**: NLP, scikit-learn, TF-IDF vectorization
- **Models**: Random Forest, SVM, or similar classifiers

### 2. Spending Analysis
- Analyze historical spending patterns
- Identify trends and seasonality
- Detect recurring expenses
- Calculate key metrics (savings rate, category percentages, etc.)

### 3. Expense Forecasting
- **Input**: Historical monthly spending data
- **Output**: Predicted total expenses + category-wise predictions
- **Metrics**: MAE, RMSE, MAPE
- **Models**: ARIMA, Prophet, XGBoost, LightGBM

### 4. Anomaly Detection
- **Input**: Current and historical transactions
- **Output**: Anomaly score, severity, explanation
- **Technology**: Isolation Forest, statistical methods, Z-score analysis
- **Approach**: Transaction-level and category-level anomalies

### 5. Budget Analysis
- Compare actual spending vs budget
- Compare predicted spending vs budget
- Generate alerts for budget violations
- Predict budget overruns

### 6. Recommendation Engine
- Combine all insights (spending, predictions, anomalies, budgets)
- Generate personalized recommendations:
  - Budget optimization
  - Savings opportunities
  - Overspending warnings
- Provide explainable reasoning for each recommendation

### 7. What-If Simulator
- Scenario modeling:
  - Income changes
  - Expense changes
  - Category budget adjustments
  - Savings target modifications
- Calculate projected financial impact
- Compare scenarios

### 8. Explainable AI (XAI)
- Explain saved Random Forest expense forecasts with global and per-user SHAP values
- Explain TF-IDF + Logistic Regression category decisions with class-specific term contributions
- Explain behavioral Isolation Forest scores without treating them as fraud probabilities
- Run the separate [SHAP explainability module](./ai-service/explainability/README.md)

## Dataset

The project uses a real personal finance dataset containing:
- ~150 users
- ~1 year of financial transaction data
- ~47,569 transactions
- Income and expense records
- Transaction categories
- Recurring payments
- Budget information
- Savings goals
- Behavioral data
- Labeled anomalies

**Important**: The original dataset is preserved in `data/database/`. Do not modify or delete it.

## Development Roadmap

### Phase 1: Foundation
- [ ] Database schema and SQLAlchemy models
- [ ] Basic API structure
- [ ] Data loading and exploration

### Phase 2: Data Pipeline
- [ ] Data preprocessing
- [ ] Transaction upload functionality
- [ ] Database integration

### Phase 3: Categorization
- [ ] Transaction categorization model
- [ ] Categorization API
- [ ] Category management

### Phase 4: Analytics
- [ ] Spending analysis calculations
- [ ] Analytics API endpoints
- [ ] Visualization queries

### Phase 5: Forecasting
- [ ] Expense forecasting model
- [ ] Forecast API
- [ ] Model evaluation

### Phase 6: Anomaly Detection
- [ ] Anomaly detection model
- [ ] Anomaly API
- [ ] Alert system

### Phase 7: Recommendations
- [ ] Recommendation engine
- [ ] Recommendation API
- [ ] Explainability system

### Phase 8: What-If Simulator
- [ ] Simulator engine
- [ ] Scenario API
- [ ] Impact calculation

### Phase 9: Dashboard & Frontend
- [ ] React/TypeScript frontend
- [ ] Dashboard components
- [ ] Real-time updates

### Phase 10: Testing & Deployment
- [ ] Unit tests
- [ ] Integration tests
- [ ] Performance optimization
- [ ] Deployment configuration

## API Endpoints (TBD)

### Authentication
- `POST /api/auth/register` - Register user
- `POST /api/auth/login` - Login
- `POST /api/auth/logout` - Logout

### Transactions
- `GET /api/transactions` - List transactions
- `POST /api/transactions` - Create transaction
- `POST /api/transactions/upload` - Upload CSV
- `GET /api/transactions/{id}` - Get transaction
- `PUT /api/transactions/{id}` - Update transaction
- `DELETE /api/transactions/{id}` - Delete transaction

### Categories
- `GET /api/categories` - List categories
- `POST /api/categories` - Create category

### Analytics
- `GET /api/analytics/summary` - Financial summary
- `GET /api/analytics/trends` - Spending trends
- `GET /api/analytics/breakdown` - Category breakdown

### Forecasts
- `GET /api/forecasts` - List forecasts
- `POST /api/forecasts/generate` - Generate forecast
- `GET /api/forecasts/accuracy` - Forecast metrics

### Anomalies
- `GET /api/anomalies` - List anomalies
- `POST /api/anomalies/detect` - Detect anomalies

### Recommendations
- `GET /api/recommendations` - List recommendations
- `POST /api/recommendations/generate` - Generate recommendations

### Simulator
- `POST /api/simulator/calculate` - Run scenario
- `GET /api/simulator/baseline` - Get baseline

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test file
pytest tests/test_transactions.py
```

## Documentation

See `docs/` folder for:
- Architecture documentation
- API specifications
- Database schema
- ML model documentation
- Deployment guide

## Key Principles

1. **Modularity**: Each component is independent and testable
2. **Separation of Concerns**: ML training separate from inference
3. **Explainability**: All recommendations include reasoning
4. **Integration**: Components work together as a system
5. **Scalability**: Design supports multiple users and datasets
6. **Data Privacy**: User data is never shared or exposed

## References & Inspirations

- Financial time-series forecasting literature
- Anomaly detection research
- Explainable AI (XAI) principles
- Personal finance app design patterns
- ML best practices for production systems

## License

This project is for educational purposes.

## Contact & Support

For issues, questions, or improvements, please refer to the documentation or create an issue.

---

**Last Updated**: 2026-09-02
**Status**: Project Structure Created - Ready for Implementation
