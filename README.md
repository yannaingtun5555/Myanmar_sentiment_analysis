# Myanmar Sentiment Analysis

Production-style Myanmar text sentiment system with Telegram intake, pipeline processing, model inference, and PostgreSQL storage.

## Pipeline

### Pipeline 1: `tg bot -> pipeline -> model -> tg bot`

1. User sends a YouTube URL to Telegram bot.
2. Bot creates a row in `requests` with status `NEW`.
3. `fetch_comment.py` downloads comments from YouTube API and stores to `raw_comments`.
4. `preprocess.py` normalizes text (Zawgyi -> Unicode, cleaning, token preparation) and stores to `preprocessed_comments`.
5. `predict_model.py` loads sentiment model and writes predictions to `predictions`.
6. `save_results.py` finalizes outputs in `final_results`.
7. Bot reads result status and returns response to the user.

### Pipeline 2: `tg bot -> pipeline -> database`

1. User sends a YouTube URL to Telegram bot.
2. Bot writes request metadata into `requests`.
3. Pipeline stages update DB tables in sequence:
   - `raw_comments` (`LOCKED`/`FETCHED`)
   - `preprocessed_comments` (`PREPROCESSED`)
   - `predictions` (`PREDICTED_MODEL`)
   - `final_results` (`FINALIZED`)
4. Low-confidence items are stored in `review_queue` for manual review.
5. Project Folder Structure
myanmar_sentiment/
│
├── dags/
│   ├── dataset_pipeline.py              # Airflow DAG for training dataset
│   └── youtube_pipeline.py              # Airflow DAG
│
├──  notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_pretokenization.ipynb
│   └── 04_db_checking.ipynb
│
├──  data/
│   ├── accuracy_test
│   ├── cleaned
│   ├── preprocessed
│   └── raw
│
├── models/
│   └── trained/
│       └── xlm-roberta-improve-final-75/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer.json
│           ├── tokenizer_config.json
│           └── label_mapping.json
│
├── src/
│   ├── db/
│   │   ├── connection.py               # DB connection (MySQL)
│   │   ├── test.py                     # DB connection test(MySQL)
│   │   └── schema.sql                  # DB schema
│   │
│   ├── pipeline/
│   │   ├── fetch_comments.py
│   │   ├── preprocess.py
│   │   ├── predict_model.py
│   │   ├── save_results.py
│   │   ├── load_dataset.py
│   │   └── load_csv.py
│   │
|   ├── models/
│   │   ├── train_base.py              # PyTorch/Hugging Face training logic
│   │   ├── accuracy_test.py           #script for accuracy testing
│   │   ├── model_dev.py               # script for refining the model
│   │   └── predict.py                 # Inference wrapper
│   │        
│   └── bot/
│       └── telegram_request_bot.py
│
├── scripts/
│   ├── load_csv_to_db.py              # labeled data import
│   └── download_dataset.py            # script for downloading dataset
│
├── configs/
│   └── config.yml
│
├── requirements.txt
└── README.md
    


## Model

- Hugging Face model: [Yanddddd/Myanmar_sentiment_model](https://huggingface.co/Yanddddd/Myanmar_sentiment_model)
- Local model paths used in this repo:
  - `models/3_class/myanmar_sentiment_final`
  - `models/7_class/xlm-roberta-improve-final-75`
- Inference script: `src/pipeline/predict_model.py`

### Benchmark (from `Banchmarks/model_result.txt`)

- Test file: `test1_pro.csv`
- Samples: `597`
- Accuracy: `0.863`
- Weighted F1-score: `0.862`

Per-class report (3-class run):
- negative: precision `0.92`, recall `0.89`, f1 `0.90`
- neutral: precision `0.78`, recall `0.95`, f1 `0.86`
- positive: precision `0.93`, recall `0.75`, f1 `0.83`

## Config Structure

File: `configs/config.yml`

```yaml
database:
  db_type: postgresql
  host: localhost
  port: 5432
  user: postgres
  password: <DB_PASSWORD>
  database: youtube_analysis

telegram:
  bot_token: <TELEGRAM_BOT_TOKEN>
  bot_token_response: <TELEGRAM_RESPONSE_BOT_TOKEN>
  allowed_users: []

youtube:
  api_key: <YOUTUBE_API_KEY>
  max_comments_per_video: 500

api_keys:
  openai_api_key: <OPENAI_API_KEY>
  huggingface_token: <HF_TOKEN>

pipeline:
  batch_size: 100
  max_comments_per_video: 1000
  confidence_threshold: 0.7
  model_path: models/3_class/myanmar_sentiment_final
```

## Necessaries

1. Python `3.10+`
2. PostgreSQL running and reachable from config.
3. Telegram bot token(s).
4. YouTube Data API v3 key.
5. Model files present locally under configured `pipeline.model_path` (or download from Hugging Face).

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run (Typical Order)

```bash
python src/pipeline/fetch_comment.py
python src/pipeline/preprocess.py
python src/pipeline/predict_model.py
python src/pipeline/save_results.py
```

Telegram bot:

```bash
python src/bot/telegram_request_bot.py
```

## Main Tables

- `requests`
- `raw_comments`
- `preprocessed_comments`
- `predictions`
- `final_results`
- `review_queue`
- `labeled_dataset`
- `training_dataset`

## Notes

- `predict_model.py` currently predicts records not yet present in `predictions`.
- Label mapping is loaded from `metadata.json` / `label_mapping.json` when available.
- For security, do not commit real API tokens in `configs/config.yml`.
