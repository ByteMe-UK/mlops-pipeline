# ⚙️ Automated ML Pipeline

An end-to-end **MLOps pipeline** — data ingestion → model training with experiment tracking → REST API serving → automated CI on every push. Built with **MLflow**, **FastAPI**, and **GitHub Actions**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-2.10+-0194E2?logo=mlflow&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/CI-GitHub_Actions-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live API

> 🔗 **[Live App →](https://mlops-pipeline-x734.onrender.com)** · **[Swagger Docs →](https://mlops-pipeline-x734.onrender.com/docs)**
>
> ⚠️ Free tier — may take 30–60s to wake up on first request.

## ✨ Features

- **Automated pipeline** — one command runs ingest → train → evaluate
- **MLflow experiment tracking** — logs params, accuracy, feature importances per run
- **Quality gate** — CI fails if test accuracy drops below 70%
- **FastAPI serving** — `POST /predict` with confidence scores and Swagger UI
- **Auto-trains on deploy** — no pre-built model files needed in the repo

## 🧠 The Problem

Predict whether a red wine is **good** (quality ≥ 6) or **bad** (quality < 6) from 11 physicochemical measurements.

**Dataset:** [UCI Wine Quality](https://archive.ics.uci.edu/ml/datasets/wine+quality) — 1 599 red wines
**Model:** Random Forest (StandardScaler + RandomForestClassifier)
**Test accuracy:** ~86%

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| scikit-learn | Random Forest pipeline |
| MLflow | Experiment tracking, metric logging |
| FastAPI | Prediction REST API |
| joblib | Model serialisation |
| GitHub Actions | CI — runs pipeline + tests API on every push |
| Render | API deployment |

## 📦 Getting Started

```bash
git clone https://github.com/ByteMe-UK/mlops-pipeline.git
cd mlops-pipeline

python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run the full pipeline (downloads data, trains, logs to MLflow)
python run_pipeline.py

# View MLflow experiment UI
mlflow ui   # open http://localhost:5000

# Start the prediction API
uvicorn api.main:app --reload
# open http://localhost:8000/docs
```

## 📡 API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed_acidity": 7.4, "volatile_acidity": 0.70,
    "citric_acid": 0.0, "residual_sugar": 1.9,
    "chlorides": 0.076, "free_sulfur_dioxide": 11.0,
    "total_sulfur_dioxide": 34.0, "density": 0.9978,
    "ph": 3.51, "sulphates": 0.56, "alcohol": 9.4
  }'
```

```json
{ "prediction": 0, "label": "bad", "confidence": 0.73, "features": { ... } }
```

## 📁 Project Structure

```
mlops-pipeline/
├── pipeline/
│   ├── ingest.py          ← Download UCI dataset, binarise target
│   └── train.py           ← Train RF, log params + metrics to MLflow
├── api/
│   └── main.py            ← FastAPI: POST /predict, GET /health
├── run_pipeline.py        ← Orchestrator + 70% accuracy quality gate
├── Procfile               ← Render deployment config
├── .github/workflows/
│   └── pipeline.yml       ← CI: run pipeline + smoke-test API on every push
├── requirements.txt
└── README.md
```

## ⚙️ How the CI Works

```
git push main
    └─▶ GitHub Actions: ubuntu-latest
            └─▶ pip install -r requirements.txt
            └─▶ python run_pipeline.py
                    └─▶ downloads wine dataset
                    └─▶ trains Random Forest (MLflow tracking)
                    └─▶ FAIL if accuracy < 70% ← quality gate
            └─▶ start uvicorn api.main:app
            └─▶ curl /health   (smoke test)
            └─▶ curl /predict  (integration test)
```

## 🚢 Deployment (Render)

1. Push repo to GitHub
2. [render.com](https://render.com) → New Web Service → connect repo
3. Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
4. Deploy — API auto-trains on first boot (~60s)

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio collection.**
