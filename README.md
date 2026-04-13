# Travel planner

LangGraph + Hugging Face Inference + FastAPI UI.

## Setup

```bash
cd travel-planner
pip install -r requirements.txt
cp .env.example .env
```

Edit **`.env`**: set **`HF_TOKEN`** (and optionally **`HF_MODEL_ID`**).

## Run web UI

```bash
python -m uvicorn travel_server:web_app --host 127.0.0.1 --port 8765 --reload
```

Open http://127.0.0.1:8765

## Run CLI

```bash
python app.py
```

See **CREDENTIALS.md** for environment variables.
