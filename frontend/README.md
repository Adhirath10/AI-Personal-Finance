# FinAI Dashboard

Responsive React + Vite dashboard connected to the existing FastAPI backend
through one Axios service layer.

## Run locally

```powershell
Copy-Item .env.example .env
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `frontend/.env` if the backend is not at
`http://localhost:8000`. Configure `FINANCE_USER_ID` in the project-root `.env`
before starting the backend. The backend and AI service startup steps,
environment variables, API routes, and response shapes are in
[docs/API_DOCUMENTATION.md](../docs/API_DOCUMENTATION.md).

## Data behavior

The dashboard does not fall back to demo values. It shows a connection or
resource error when an API request fails. The visible data comes from the
backend's configured user profile and the configured SQLite database; AI
results appear only after a corresponding model endpoint responds. Stale
forecast amounts are withheld, and anomaly detections are described as
unusual transactions requiring review, not confirmed fraud.

Profile preferences in Settings are saved in browser local storage. Account
profile, authentication, and security APIs are not implemented.

## Structure

- `src/components/`: reusable navigation, cards, charts, and tables
- `src/pages/Pages.jsx`: routed dashboard pages
- `src/services/api.js`: Axios client and backend resource/action mappings
- `src/utils/format.js`: INR and date formatting
