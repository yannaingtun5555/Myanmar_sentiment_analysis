#!/usr/bin/env python3
import argparse
import logging
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import yaml
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnlabeledExporter:
    """Export comments that need manual labeling"""
    
    def __init__(self, config_path: str = '../configs/config.yaml'):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        db_config = self.config['database']
        db_url = f"postgresql://{db_config['user']}:{db_config['password']}@{db_config['host']}:{db_config['port']}/{db_config['database']}"
        self.engine = create_engine(db_url)
    
    def export_review_queue(self, output_path: str, limit: int = 1000):
        """Export pending reviews from review_queue"""
        query = f"""
            SELECT 
                rq.id,
                rq.comment_id,
                rq.req_id,
                rq.text,
                rq.model_prediction,
                rq.ai_prediction,
                rq.confidence,
                rc.text_unicode as cleaned_text
            FROM review_queue rq
            LEFT JOIN preprocessed_comments rc ON rq.comment_id = rc.comment_id
            WHERE rq.status = 'PENDING'
            ORDER BY rq.confidence ASC
            LIMIT {limit}
        """
        
        df = pd.read_sql(query, self.engine)
        
        # Add column for manual label
        df['manual_label'] = ''
        
        # Select relevant columns
        export_cols = ['comment_id', 'text', 'cleaned_text', 'model_prediction', 
                      'ai_prediction', 'confidence', 'manual_label']
        df_export = df[export_cols]
        
        # Export to CSV
        df_export.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df_export)} comments to {output_path}")
        
        return df_export
    
    def export_low_confidence(self, output_path: str, threshold: float = 0.7, limit: int = 1000):
        """Export low confidence predictions for review"""
        query = f"""
            SELECT 
                p.comment_id,
                p.req_id,
                p.model_prediction,
                p.model_confidence,
                p.ai_prediction,
                p.ai_confidence,
                pc.text_unicode as cleaned_text
            FROM predictions p
            JOIN preprocessed_comments pc ON p.comment_id = pc.comment_id
            WHERE p.model_confidence < {threshold} 
               OR p.ai_confidence < {threshold}
            LIMIT {limit}
        """
        
        df = pd.read_sql(query, self.engine)
        df['manual_label'] = ''
        
        df.to_csv(output_path, index=False)
        logger.info(f"Exported {len(df)} low-confidence comments to {output_path}")
        
        return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--export-type', choices=['review_queue', 'low_confidence'], 
                       default='review_queue')
    parser.add_argument('--output', type=str, default='unlabeled_comments.csv')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=0.7)
    
    args = parser.parse_args()
    
    exporter = UnlabeledExporter()
    
    if args.export_type == 'review_queue':
        exporter.export_review_queue(args.output, args.limit)
    else:
        exporter.export_low_confidence(args.output, args.threshold, args.limit)