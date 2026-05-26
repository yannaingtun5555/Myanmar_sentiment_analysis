# Myanmar YouTube Sentiment Analysis

Production-style pipeline for **Myanmar-language YouTube comment sentiment analysis** using **XLM-RoBERTa** with Telegram request handling and PostgreSQL-backed processing.

## Overview

This project:
- accepts YouTube links from a Telegram bot,
- fetches comments via YouTube Data API,
- normalizes Myanmar text (including Zawgyi -> Unicode),
- predicts one of 7 emotion classes,
- saves final results and low-confidence cases for human review.

Emotion classes:
- `Anger`
- `Fear`
- `Joy`
- `postitive`
- `Neutral`
- `Sadness`
- `Surprise`

Final selected model performance:
- **Weighted F1: 0.75**

## Architecture

Source: `structure/archi.txt`

### End-to-end flow

1. User sends YouTube URL to Telegram bot.
2. Request is inserted into `requests` (`NEW`).
3. `fetch_comment.py` gets comments from YouTube API and stores in `raw_comments`.
4. `preprocess.py` converts/cleans/tokenizes text into `preprocessed_comments`.
5. `predict_model.py` loads model and writes predictions to `predictions`.
6. `save_results.py`:
- writes normal-confidence outputs to `final_results`,
- writes low-confidence outputs to `review_queue`,
- sends Telegram completion notification.

### Request status lifecycle

`NEW -> LOCKED -> FETCHED -> PREPROCESSED -> PREDICTED_MODEL -> FINALIZED -> RESPONDED`

## Project Structure

```text
myanmar_sentiment/
├── configs/
│   └── config.yml
├── data/
│   ├── raw/
│   ├── cleaned/
│   ├── processed/
│   └── annotated/
├── models/
│   └── trained/
│       └── xlm-roberta-improve-final-75/
├── notebooks/
├── scripts/
│   ├── load_csv_to_db.py
│   └── download_datasets.py
├── src/
│   ├── bot/
│   │   └── telegram_request_bot.py
│   ├── db/
│   │   ├── connection.py
│   │   └── schema.sql
│   ├── models/
│   │   ├── train_base.py
│   │   ├── model_dev.py
│   │   ├── model_loader.py
│   │   └── predict.py
│   └── pipeline/
│       ├── fetch_comment.py
│       ├── preprocess.py
│       ├── predict_model.py
│       ├── save_results.py
│       └── load_csv.py
└── test_model.py
```

## Database Design

Main tables:
- `requests`
- `raw_comments`
- `preprocessed_comments`
- `predictions`
- `final_results`
- `review_queue`
- `labeled_dataset`
- `training_dataset`

Schema file: `src/db/schema.sql`

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure project

Edit:

```bash
configs/config.yml
```

Required sections:
- `database`
- `telegram.bot_token`
- `telegram.bot_token_response`
- `youtube.api_key`
- `pipeline.model_path`

Example `configs/config.yml` structure:

```yaml
database:
  db_type: postgresql
  host: localhost
  port: 5432
  user: your_db_user
  password: your_db_password
  database: youtube_analysis

telegram:
  bot_token: "YOUR_TELEGRAM_REQUEST_BOT_TOKEN"
  bot_token_response: "YOUR_TELEGRAM_RESPONSE_BOT_TOKEN"
  allowed_users: []

youtube:
  api_key: "YOUR_YOUTUBE_DATA_API_KEY"
  max_comments_per_video: 500

api_keys:
  openai_api_key: "YOUR_OPENAI_API_KEY"
  huggingface_token: "YOUR_HUGGINGFACE_TOKEN"

pipeline:
  batch_size: 100
  max_comments_per_video: 1000
  confidence_threshold: 0.7
  model_path: "models/trained/xlm-roberta-improve-final-75"
```

If `configs/config.yml` is gitignored, create it locally with the template above.

### 3. Initialize database

Run your schema SQL in PostgreSQL:

```bash
psql -U <user> -d <db_name> -f src/db/schema.sql
```

## Running the System

### Start Telegram request bot

```bash
python src/bot/telegram_request_bot.py
```

### Run batch pipeline manually

```bash
python src/pipeline/fetch_comment.py
python src/pipeline/preprocess.py
python src/pipeline/predict_model.py
python src/pipeline/save_results.py
```

## Model Development (5 Training Rounds)

The final model (`xlm-roberta-improve-final-75`) is the result of **5 iterative training rounds**.

### Round 1
- Baseline fine-tuning with standard train/validation split.

### Round 2
- Label cleanup and normalization (e.g., spelling/format fixes).

### Round 3
- Class imbalance handling with weighted loss.

### Round 4
- Data expansion/combination with additional cleaned samples.

### Round 5 (Final)
- Hyperparameter and data refinements consolidated into final run.
- Selected production artifact: `models/trained/xlm-roberta-improve-final-75`
- **Final weighted F1 = 0.75**

## Inference Model Path

Configured at:

```yaml
pipeline:
  model_path: models/trained/xlm-roberta-improve-final-75
```

## Notes

- Low-confidence predictions (`< 0.6`) are sent to `review_queue` for human labeling.
- After finalization, users receive Telegram notifications with emotion summary.
- Current `config.yml` contains live-looking tokens/keys; rotate and move secrets to environment variables for safety.
