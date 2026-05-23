-- =====================================================
-- REQUESTS TABLE
-- =====================================================
CREATE TYPE request_status AS ENUM (
    'NEW',
    'LOCKED',
    'FETCHED',
    'PREPROCESSED',
    'PREDICTED_MODEL',
    'FINALIZED',
    'RESPONDED',
    'FAILED'
);

CREATE TABLE IF NOT EXISTS requests (
    req_id          SERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL,
    video_url       TEXT NOT NULL,
    status          request_status DEFAULT 'NEW',
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_requests_status ON requests(status);
CREATE INDEX IF NOT EXISTS idx_requests_created_at ON requests(created_at);

-- =====================================================
-- RAW_COMMENTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS raw_comments (
    comment_id      VARCHAR(50) PRIMARY KEY,
    video_id        VARCHAR(20),
    req_id          INT REFERENCES requests(req_id) ON DELETE CASCADE,
    text_display    TEXT,
    scraped_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_raw_comments_req_id ON raw_comments(req_id);
CREATE INDEX IF NOT EXISTS idx_raw_comments_video_id ON raw_comments(video_id);

-- =====================================================
-- PREPROCESSED_COMMENTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS preprocessed_comments (
    comment_id      VARCHAR(50) PRIMARY KEY,
    req_id          INT NOT NULL,
    text_unicode    TEXT,
    tokens          TEXT[],
    processed_by    VARCHAR(30),
    processed_at    TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_preprocessed_req_id ON preprocessed_comments(req_id);

-- =====================================================
-- PREDICTIONS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS predictions (
    id               SERIAL PRIMARY KEY,
    comment_id       VARCHAR(50),
    req_id           INT,
    model_prediction VARCHAR(20),
    model_confidence FLOAT,
    created_at       TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_comment_id ON predictions(comment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_req_id ON predictions(req_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model ON predictions(model_prediction);

-- =====================================================
-- FINAL_RESULTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS final_results (
    id              SERIAL PRIMARY KEY,
    comment_id      VARCHAR(50),
    req_id          INT,
    emotion_class   VARCHAR(20),
    source          VARCHAR(20),
    created_at      TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_final_results_req_id ON final_results(req_id);
CREATE INDEX IF NOT EXISTS idx_final_results_emotion ON final_results(emotion_class);

-- =====================================================
-- REVIEW_QUEUE TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS review_queue (
    id               SERIAL PRIMARY KEY,
    comment_id       VARCHAR(50),
    req_id           INT,
    text             TEXT,
    model_prediction VARCHAR(20),
    confidence       FLOAT,
    status           VARCHAR(20) DEFAULT 'PENDING',
    created_at       TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_review_queue_status ON review_queue(status);
CREATE INDEX IF NOT EXISTS idx_review_queue_confidence ON review_queue(confidence) WHERE status = 'PENDING';

-- =====================================================
-- LABELED_DATASET TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS labeled_dataset (
    id              SERIAL PRIMARY KEY,
    comment_id      VARCHAR(50),
    req_id          INT,
    comment_text    TEXT,
    label           VARCHAR(20),
    predicited      VARCHAR(20),
    labeled_by      VARCHAR(50),
    labeled_at      TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_labeled_dataset_label ON labeled_dataset(label);
CREATE INDEX IF NOT EXISTS idx_labeled_dataset_comment_id ON labeled_dataset(comment_id);

-- =====================================================
-- TRAINING_DATASET TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS training_dataset (
    id              SERIAL PRIMARY KEY,
    comment_text    TEXT NOT NULL,
    label           VARCHAR(20) NOT NULL,
    source          VARCHAR(20),
    source_id       INT,
    confidence      FLOAT,
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_training_dataset_label ON training_dataset(label);
CREATE INDEX IF NOT EXISTS idx_training_dataset_source ON training_dataset(source);

-- =====================================================
-- VIEW: Pending Reviews
-- =====================================================
CREATE OR REPLACE VIEW v_pending_reviews AS
SELECT 
    rq.id,
    rq.comment_id,
    rq.req_id,
    rq.text,
    rq.model_prediction,
    rq.confidence,
    rc.video_id,
    req.user_id,
    req.video_url,
    rq.created_at
FROM review_queue rq
LEFT JOIN raw_comments rc ON rq.comment_id = rc.comment_id
LEFT JOIN requests req ON rq.req_id = req.req_id
WHERE rq.status = 'PENDING'
ORDER BY rq.created_at;

-- =====================================================
-- VIEW: Training Data Export
-- =====================================================
CREATE OR REPLACE VIEW v_training_data_export AS
SELECT 
    comment_text,
    label,
    source,
    confidence,
    created_at
FROM training_dataset
WHERE label IS NOT NULL
ORDER BY created_at DESC;