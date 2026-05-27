#!/usr/bin/env python3
import os
import yaml
import torch
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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
    
    if model_path.startswith('.'):
        abs_path = os.path.join(project_root, model_path)
    elif model_path.startswith('/'):
        abs_path = model_path
    else:
        abs_path = os.path.join(project_root, model_path)
    
    return os.path.normpath(abs_path)

def get_db():
    config = load_config()
    db = config['database']
    return psycopg2.connect(
        host=db['host'], port=db['port'],
        database=db['database'], user=db['user'], password=db['password']
    )

def load_label_mapping(model_path):
    """Load label mapping from model metadata or config"""
    # Try to load from metadata.json first
    metadata_path = os.path.join(model_path, 'metadata.json')
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            if 'id2label' in metadata:
                id2label = {int(k): v for k, v in metadata['id2label'].items()}
                print(f"✅ Loaded label mapping from metadata.json")
                return id2label
    
    # Try label_mapping.json
    label_mapping_path = os.path.join(model_path, 'label_mapping.json')
    if os.path.exists(label_mapping_path):
        with open(label_mapping_path, 'r') as f:
            mapping = json.load(f)
            if 'id2label' in mapping:
                id2label = {int(k): v for k, v in mapping['id2label'].items()}
                print(f"✅ Loaded label mapping from label_mapping.json")
                return id2label
    
    # Try to get from model config
    print("⚠️ No label mapping file found, trying model config...")
    return None

def load_model():
    model_path = get_model_path()
    print(f"Loading model from: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
    
    # Load label mapping
    id2label = load_label_mapping(model_path)
    
    if id2label is None:
        # Try to get from model config
        if hasattr(model.config, 'id2label') and model.config.id2label:
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            print(f"✅ Loaded label mapping from model config")
        else:
            # Default for 3 classes
            print("⚠️ Using default label mapping (3 classes)")
            id2label = {0: "positive", 1: "negative", 2: "neutral"}
    
    print(f"📋 Label mapping: {id2label}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    return tokenizer, model, device, id2label

def predict(text, tokenizer, model, device, id2label):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    
    # Use label mapping from model
    predicted_label = id2label.get(pred_id, f"class_{pred_id}")
    
    return predicted_label, confidence

def main():
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT pc.comment_id, pc.text_unicode, pc.req_id
            FROM preprocessed_comments pc
            INNER JOIN raw_comments rc ON pc.comment_id = rc.comment_id
            LEFT JOIN predictions p ON pc.comment_id = p.comment_id
            WHERE p.comment_id IS NULL
            LIMIT 500
        """)
        comments = cur.fetchall()
        
        if not comments:
            print("No comments to predict")
            return
        
        print(f"Loading model from {get_model_path()}")
        tokenizer, model, device, id2label = load_model()
        print(f"Using device: {device}")
        print(f"Predicting {len(comments)} comments")
        
        predicted = 0
        req_ids = set()
        
        for c in comments:
            label, confidence = predict(c['text_unicode'], tokenizer, model, device, id2label)
            
            cur.execute("""
                INSERT INTO predictions (comment_id, req_id, model_prediction, model_confidence, created_at)
                VALUES (%s, %s, %s, %s, NOW())
            """, (c['comment_id'], c['req_id'], label, confidence))
            predicted += 1
            req_ids.add(c['req_id'])
            conn.commit()
            
            if predicted % 10 == 0:
                print(f"Predicted {predicted} comments")
        
        # Update request status to PREDICTED
        for req_id in req_ids:
            cur.execute("""
                UPDATE requests 
                SET status = 'PREDICTED_MODEL' 
                WHERE req_id = %s AND status = 'FETCHED'
            """, (req_id,))
        
        conn.commit()
        print(f"Done. Predicted: {predicted}")
        print(f"Updated requests: {len(req_ids)} to status PREDICTED")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()