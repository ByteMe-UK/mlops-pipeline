# ⚙️ Automated ML Pipeline (MLOps)

An end-to-end **MLOps pipeline** — from data ingestion to model training, evaluation, and deployment — fully automated with **GitHub Actions**.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-tracking-0194E2?logo=mlflow&logoColor=white)
![GitHub Actions](https://img.shields.io/badge/GitHub_Actions-CI/CD-2088FF?logo=githubactions&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

## 🚀 Live Demo

> 🔗 _Coming soon — API deployed on Render, pipeline on GitHub Actions_

## 📸 Screenshots

> _Screenshots will be added after the pipeline is built_

## ✨ Features

- Automated data ingestion and preprocessing
- Model training with experiment tracking (MLflow)
- Model evaluation with metrics logging
- Automated deployment via GitHub Actions CI/CD
- FastAPI serving endpoint for predictions

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10+ | Core language |
| DVC | Data version control |
| MLflow | Experiment tracking |
| Scikit-learn | Model training |
| FastAPI | Model serving API |
| GitHub Actions | CI/CD automation |

## 📦 Getting Started

```bash
# Clone the repo
git clone https://github.com/ByteMe-UK/mlops-pipeline.git
cd mlops-pipeline

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline
python pipeline/run.py
```

## 📁 Project Structure

```
mlops-pipeline/
├── pipeline/
│   ├── run.py             ← Pipeline orchestrator
│   ├── ingest.py          ← Data ingestion
│   ├── train.py           ← Model training
│   ├── evaluate.py        ← Model evaluation
│   └── deploy.py          ← Deployment step
├── api/
│   └── main.py            ← FastAPI prediction endpoint
├── .github/workflows/     ← CI/CD pipeline
├── requirements.txt
└── README.md
```

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

**Part of the [ByteMe-UK](https://github.com/ByteMe-UK) portfolio collection.**
