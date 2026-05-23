#!/usr/bin/env python3
import os
import csv
import yaml
import psycopg2
from datetime import datetime

def load_config():
    with open('configs/config.yml', 'r') as f:
        return yaml.safe_load(f)

def get_db():
    config = load_config()
    db = config['database']
    return psycopg2.connect(
        host=db['host'], port=db['port'],
        database=db['database'], user=db['user'], password=db['password']
    )

def export_review_queue():
    conn = get_db()
    cur = conn.cursor()
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    # Create data/annotated directory if not exists
    output_dir = os.path.join(project_root, 'data', 'annotated')
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f'review_queue_{timestamp}.csv')
    
    cur.execute("""
        SELECT 
            rq.req_id,
            rq.comment_id,
            rq.text,
            rq.model_prediction,
            rq.confidence,
            rq.status,
            rq.created_at
        FROM review_queue rq
        WHERE rq.status = 'PENDING'
        ORDER BY rq.confidence, rq.created_at
    """)
    
    rows = cur.fetchall()
    
    if not rows:
        print("No pending reviews in queue")
        cur.close()
        conn.close()
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['req_id', 'comment_id', 'text', 'prediction', 'confidence', 'status', 'label'])
        
        for row in rows:
            writer.writerow([row[0], row[1], row[2], row[3], row[4], row[5], ''])
    
    print(f"Exported {len(rows)} comments to: {output_file}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    export_review_queue()