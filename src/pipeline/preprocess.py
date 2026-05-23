#!/usr/bin/env python3
import os
import re
import sys
import yaml
import psycopg2
from psycopg2.extras import RealDictCursor
from Siamese import BurmeseConverter

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    project_root = get_project_root()
    config_path = os.path.join(project_root, 'configs', 'config.yml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model_path():
    config = load_config()
    model_path = config['pipeline']['model_path']
    project_root = get_project_root()
    
    if model_path.startswith('..'):
        abs_path = os.path.join(project_root, model_path)
    elif model_path.startswith('/'):
        abs_path = model_path
    else:
        abs_path = os.path.join(project_root, model_path)
    
    return os.path.normpath(abs_path)

_converter = None
_tokenizer = None

def get_converter():
    global _converter
    if _converter is None:
        _converter = BurmeseConverter()
    return _converter

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        from transformers import AutoTokenizer
        model_path = get_model_path()
        _tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    return _tokenizer

def get_db():
    config = load_config()
    db = config['database']
    return psycopg2.connect(
        host=db['host'], port=db['port'],
        database=db['database'], user=db['user'], password=db['password']
    )

def is_burmese(text):
    burmese_range = re.compile(r'[\u1000-\u109F]')
    matches = burmese_range.findall(text)
    if len(matches) == 0:
        return False
    burmese_ratio = len(matches) / len(text) if len(text) > 0 else 0
    return burmese_ratio > 0.3

def zawgyi_to_unicode(text):
    try:
        converter = get_converter()
        return converter.zawgyi_to_unicode(text)
    except Exception as e:
        print(f"Conversion error: {e}")
        return text

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^\u1000-\u109F\s\.\,\!\\?\(\)]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_with_roberta(text):
    try:
        tokenizer = get_tokenizer()
        tokens = tokenizer.tokenize(text)
        return tokens
    except Exception as e:
        print(f"Tokenizer error: {e}")
        return text.split()

def preprocess_comment(text):
    if not is_burmese(text):
        return None, None, None
    text = zawgyi_to_unicode(text)
    text = clean_text(text)
    if len(text) < 5:
        return None, None, None
    tokens = tokenize_with_roberta(text)
    return text, tokens, len(tokens)

def main():
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT rc.comment_id, rc.text_display, rc.req_id
            FROM raw_comments rc
            LEFT JOIN preprocessed_comments pc ON rc.comment_id = pc.comment_id
            WHERE pc.comment_id IS NULL
            AND rc.req_id IN (
                SELECT req_id FROM requests WHERE status = 'FETCHED'
            )
            LIMIT 500
        """)
        comments = cur.fetchall()
        
        if not comments:
            print("No comments to preprocess")
            return
        
        print(f"Processing {len(comments)} comments")
        processed = 0
        skipped = 0
        
        for c in comments:
            unicode_text, tokens, token_len = preprocess_comment(c['text_display'])
            
            if unicode_text is None:
                skipped += 1
                continue
            
            cur.execute("""
                INSERT INTO preprocessed_comments 
                (comment_id, req_id, text_unicode, tokens, processed_by, processed_at)
                VALUES (%s, %s, %s, %s, 'base_model', NOW())
                ON CONFLICT (comment_id) DO UPDATE
                SET text_unicode = EXCLUDED.text_unicode,
                    tokens = EXCLUDED.tokens,
                    processed_at = NOW()
            """, (c['comment_id'], c['req_id'], unicode_text, tokens))
            processed += 1
            
            if processed % 50 == 0:
                conn.commit()
                print(f"Processed {processed} comments")
        
        conn.commit()
        print(f"Done. Processed: {processed}, Skipped: {skipped}")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()