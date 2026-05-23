#!/usr/bin/env python3
"""
Save predictions to final_results and review_queue
Then notify users via separate Telegram notification bot
"""

import os
import sys
import yaml
import psycopg2
from psycopg2.extras import RealDictCursor
import requests as http_requests
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

def send_telegram_message(bot_token, chat_id, message):
    """Send plain text message via Telegram Bot API"""
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        'chat_id': chat_id,
        'text': message,
        # Remove parse_mode entirely - don't send it
    }
    
    try:
        response = http_requests.post(url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get('ok'):
                print(f"  ✅ Notification sent to {chat_id}")
                return True
        print(f"  ❌ Failed: {response.text[:200]}")
        return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def get_request_stats(conn, req_id):
    """Get statistics for a request"""
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                emotion_class,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) as percentage
            FROM final_results
            WHERE req_id = %s
            GROUP BY emotion_class
            ORDER BY count DESC
            LIMIT 1
        """, (req_id,))
        top_result = cur.fetchone()
        
        cur.execute("SELECT COUNT(*) as total FROM final_results WHERE req_id = %s", (req_id,))
        total = cur.fetchone()['total']
        
        # Get all emotions for full breakdown
        cur.execute("""
            SELECT emotion_class, COUNT(*) as count
            FROM final_results
            WHERE req_id = %s
            GROUP BY emotion_class
            ORDER BY count DESC
        """, (req_id,))
        all_emotions = cur.fetchall()
        
        if top_result:
            return {
                'top_emotion': top_result['emotion_class'],
                'top_count': top_result['count'],
                'top_percent': top_result['percentage'],
                'total_comments': total,
                'all_emotions': all_emotions
            }
        return None

def main():
    config = load_config()
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    # Use the RESPONSE bot token for notifications
    bot_token = config['telegram']['bot_token_response']
    
    print(f"\n🔧 Using Notification Bot Token: {bot_token[:15]}...")
    
    # Test notification bot connection
    test_url = f"https://api.telegram.org/bot{bot_token}/getMe"
    try:
        test_response = http_requests.get(test_url, timeout=5)
        if test_response.status_code == 200:
            bot_info = test_response.json()
            username = bot_info.get('result', {}).get('username', 'unknown')
            print(f"✅ Notification bot connected: @{username}")
        else:
            print(f"❌ Notification bot connection failed! Status: {test_response.status_code}")
            return
    except Exception as e:
        print(f"❌ Cannot connect to notification bot: {e}")
        return
    
    # Emotion emoji mapping
    emoji_map = {
        'anger': '😠', 'fear': '😨', 'joy': '😊',
        'love': '❤️', 'neutral': '😐', 'sadness': '😢', 'surprise': '😲'
    }
    
    try:
        # Get all requests that are PREDICTED_MODEL but not yet FINALIZED
        cur.execute("""
            SELECT DISTINCT p.req_id, r.user_id
            FROM predictions p
            JOIN requests r ON p.req_id = r.req_id
            WHERE r.status = 'PREDICTED_MODEL'
            AND p.model_prediction IS NOT NULL
        """)
        requests = cur.fetchall()
        
        if not requests:
            print("No requests to process")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing {len(requests)} request(s)")
        print(f"{'='*60}")
        
        for req in requests:
            req_id = req['req_id']
            user_id = req['user_id']
            print(f"\n📋 Processing request #{req_id} (User: {user_id})")
            
            # Get predictions for this request
            cur.execute("""
                SELECT p.comment_id, p.model_prediction, p.model_confidence, rc.text_display
                FROM predictions p
                JOIN raw_comments rc ON p.comment_id = rc.comment_id
                WHERE p.req_id = %s
            """, (req_id,))
            predictions = cur.fetchall()
            
            if not predictions:
                print(f"  ⚠️ No predictions found for request {req_id}")
                continue
            
            # Insert into final_results
            inserted = 0
            for pred in predictions:
                try:
                    cur.execute("""
                        INSERT INTO final_results (comment_id, req_id, emotion_class, source, created_at)
                        VALUES (%s, %s, %s, 'MODEL', NOW())
                    """, (pred['comment_id'], req_id, pred['model_prediction']))
                    inserted += 1
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    continue
            
            print(f"  ✅ Added {inserted} records to final_results")
            
            # Insert low confidence predictions into review_queue (confidence < 0.6)
            low_conf = [p for p in predictions if p['model_confidence'] < 0.6]
            
            for pred in low_conf:
                try:
                    cur.execute("""
                        INSERT INTO review_queue (comment_id, req_id, text, model_prediction, confidence, created_at)
                        VALUES (%s, %s, %s, %s, %s, NOW())
                    """, (pred['comment_id'], req_id, pred['text_display'], 
                          pred['model_prediction'], pred['model_confidence']))
                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    continue
            
            print(f"  📝 Added {len(low_conf)} low-confidence comments to review_queue")
            
            # Update request status to FINALIZED
            cur.execute("""
                UPDATE requests 
                SET status = 'FINALIZED' 
                WHERE req_id = %s
            """, (req_id,))
            
            conn.commit()
            print(f"  🔄 Request {req_id} status updated to FINALIZED")
            
            # Get statistics for notification
            stats = get_request_stats(conn, req_id)
            
            # Send Telegram notification via RESPONSE bot
            if stats and user_id:
                print(f"\n  📤 Sending notification via RESPONSE bot to user {user_id}...")
                
                # Build plain text message (no markdown)
                top_emoji = emoji_map.get(stats['top_emotion'], '📊')
                
                # Create a clean, simple message
                message_lines = [
                    "✅ Analysis Complete!",
                    "",
                    f"Request #{req_id} has been processed.",
                    "",
                    f"📝 {stats['total_comments']} comments analyzed",
                    f"{top_emoji} Top emotion: {stats['top_emotion']} ({stats['top_count']} comments, {stats['top_percent']}%)",
                    "",
                    "📊 Full breakdown:"
                ]
                
                for emo in stats['all_emotions']:
                    emoji = emoji_map.get(emo['emotion_class'], '•')
                    message_lines.append(f"{emoji} {emo['emotion_class']}: {emo['count']} comments")
                
                message_lines.extend([
                    "",
                    f"Use /status_{req_id} to view detailed results."
                ])
                
                message = "\n".join(message_lines)
                
                print(f"  Message length: {len(message)} chars")
                
                # Send using response bot (no parse_mode parameter)
                success = send_telegram_message(bot_token, user_id, message)
                
                if success:
                    print(f"  ✅ Notification sent successfully!")
                    # Update status to RESPONDED
                    cur.execute("""
                        UPDATE requests 
                        SET status = 'RESPONDED' 
                        WHERE req_id = %s
                    """, (req_id,))
                    conn.commit()
                    print(f"  📝 Request {req_id} status updated to RESPONDED")
                else:
                    print(f"  ❌ Failed to send notification")
            else:
                print(f"  ⚠️ No stats or user_id for notification")
        
        print(f"\n{'='*60}")
        print("✅ All requests processed successfully!")
        print(f"{'='*60}\n")
        
        # Show summary
        cur.execute("""
            SELECT status, COUNT(*) as count
            FROM requests
            GROUP BY status
        """)
        summary = cur.fetchall()
        print("📊 Database Summary:")
        for row in summary:
            print(f"  {row['status']}: {row['count']} requests")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        conn.rollback()
    finally:
        cur.close()
        conn.close()
        print("\n🏁 Script finished")

if __name__ == "__main__":
    main()