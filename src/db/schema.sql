-- =====================================================
-- REQUESTS TABLE
-- =====================================================
CREATE TYPE request_status AS ENUM (
    'NEW',
    'LOCKED',
    'FETCHIED',
    'PREPROCESSED',
    'PREDICTED_MODEL',
    'PREDICTED_AI',
    'COMPARED',
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
    ai_prediction    VARCHAR(20),
    ai_confidence    FLOAT,
    created_at       TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_predictions_comment_id ON predictions(comment_id);
CREATE INDEX IF NOT EXISTS idx_predictions_req_id ON predictions(req_id);
CREATE INDEX IF NOT EXISTS idx_predictions_model_ai ON predictions(model_prediction, ai_prediction);

-- =====================================================
-- COMPARISON_RESULTS TABLE
-- =====================================================
CREATE TABLE IF NOT EXISTS comparison_results (
    id               SERIAL PRIMARY KEY,
    comment_id       VARCHAR(50),
    req_id           INT,
    decision         VARCHAR(20),
    final_class      VARCHAR(20),
    created_at       TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_comparison_decision ON comparison_results(decision);
CREATE INDEX IF NOT EXISTS idx_comparison_req_id ON comparison_results(req_id);

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
    ai_prediction    VARCHAR(20),
    confidence       FLOAT,
    label            VARCHAR(20),
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
    comment_text    TEXT,
    label           VARCHAR(20),
    source          VARCHAR(20),
    req_id          INT,
    comment_id      VARCHAR(50),
    created_at      TIMESTAMP DEFAULT NOW(),
    FOREIGN KEY (comment_id) REFERENCES raw_comments(comment_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_labeled_dataset_label ON labeled_dataset(label);
CREATE INDEX IF NOT EXISTS idx_labeled_dataset_source ON labeled_dataset(source);
CREATE INDEX IF NOT EXISTS idx_labeled_dataset_created ON labeled_dataset(created_at);

-- =====================================================
-- VIEWS
-- =====================================================
CREATE OR REPLACE VIEW v_pending_reviews AS
SELECT 
    rq.id,
    rq.comment_id,
    rq.req_id,
    rq.text,
    rq.model_prediction,
    rq.ai_prediction,
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

CREATE OR REPLACE VIEW v_disagreements AS
SELECT 
    p.comment_id,
    p.model_prediction,
    p.model_confidence,
    p.ai_prediction,
    p.ai_confidence,
    cr.decision,
    cr.final_class,
    rc.text_display,
    req.user_id
FROM predictions p
LEFT JOIN comparison_results cr ON p.comment_id = cr.comment_id
LEFT JOIN raw_comments rc ON p.comment_id = rc.comment_id
LEFT JOIN requests req ON p.req_id = req.req_id
WHERE cr.decision IN ('MISMATCH', 'LOW_CONFIDENCE')
ORDER BY cr.created_at DESC;

CREATE OR REPLACE VIEW v_training_data_export AS
SELECT 
    comment_text,
    label,
    source,
    created_at
FROM labeled_dataset
WHERE label IS NOT NULL
ORDER BY created_at DESC;