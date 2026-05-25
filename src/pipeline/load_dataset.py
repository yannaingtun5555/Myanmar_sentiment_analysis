#!/usr/bin/env python3
import os
import csv
import yaml
import psycopg2
from datetime import datetime

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def load_config():
    config_path = os.path.join(get_project_root(), 'configs', 'config.yml')
    with open(config_path, 'r', encoding='utf-8') as f:
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
    
    project_root = get_project_root()
    
    # Create data/annotated directory if not exists
    output_dir = os.path.join(project_root, 'data', 'accuracy_test')  # Fixed typo
    os.makedirs(output_dir, exist_ok=True)
    
    # Output file with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')  # Fixed: actually use timestamp format
    output_file = os.path.join(output_dir, f'review_queue_{timestamp}.csv')
    
    # Fixed query - added proper table reference and removed alias
    cur.execute(
        """
        SELECT rc.req_id, rc.text_display
        FROM raw_comments rc
        JOIN requests r ON r.req_id = rc.req_id
        WHERE r.status = 'FETCHED'
        ORDER BY rc.scraped_at
        """
    )
    
    rows = cur.fetchall()
    
    if not rows:
        print("No pending reviews in queue")
        cur.close()
        conn.close()
        return
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'label'])
        
        req_ids = set()
        for req_id, text_display in rows:
            writer.writerow([text_display, ''])
            if req_id is not None:
                req_ids.add(req_id)

    if req_ids:
        cur.execute(
            """
            UPDATE requests
            SET status = 'FAILED'
            WHERE req_id = ANY(%s)
            """,
            (list(req_ids),),
        )
        conn.commit()
        print(f"Updated {len(req_ids)} request(s) to FAILED")
    
    print(f"Exported {len(rows)} comments to: {output_file}")
    
    cur.close()
    conn.close()

if __name__ == "__main__":
    export_review_queue()
