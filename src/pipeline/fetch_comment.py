import os
import sys
import time
import yaml
import psycopg2
from psycopg2.extras import RealDictCursor
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from urllib.parse import urlparse, parse_qs

def load_config():
    with open('configs/config.yml', 'r') as f:
        return yaml.safe_load(f)

def get_db():
    config = load_config()
    db = config['database']
    return psycopg2.connect(
        host=db['host'],
        port=db['port'],
        database=db['database'],
        user=db['user'],
        password=db['password']
    )

def extract_video_id(url):
    parsed = urlparse(url)
    if 'youtu.be' in parsed.netloc:
        return parsed.path.lstrip('/')
    if 'youtube.com' in parsed.netloc:
        if 'shorts' in parsed.path:
            return parsed.path.split('/')[-1]
        query = parse_qs(parsed.query)
        return query.get('v', [None])[0]
    return None

def fetch_comments(video_id, api_key, max_results=500):
    youtube = build('youtube', 'v3', developerKey=api_key)
    comments = []
    next_token = None
    
    while len(comments) < max_results:
        try:
            request = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                maxResults=min(100, max_results - len(comments)),
                pageToken=next_token,
                textFormat='plainText'
            )
            response = request.execute()
            
            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']
                comments.append({
                    'id': item['snippet']['topLevelComment']['id'],
                    'text': comment['textDisplay']
                })
            
            next_token = response.get('nextPageToken')
            if not next_token:
                break
                
        except HttpError as e:
            print(f"YouTube API error: {e}")
            break
    
    return comments

def main():
    config = load_config()
    api_key = config['youtube']['api_key']
    max_comments = config['youtube'].get('max_comments_per_video', 500)
    
    conn = get_db()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute("""
            SELECT req_id, video_url 
            FROM requests 
            WHERE status = 'NEW' 
            ORDER BY created_at 
            LIMIT 1 
            FOR UPDATE SKIP LOCKED
        """)
        request = cur.fetchone()
        
        if not request:
            print("No pending requests")
            return
        
        req_id = request['req_id']
        video_url = request['video_url']
        video_id = extract_video_id(video_url)
        
        if not video_id:
            print(f"Invalid video URL: {video_url}")
            cur.execute("UPDATE requests SET status = 'FAILED' WHERE req_id = %s", (req_id,))
            conn.commit()
            return
        
        cur.execute("UPDATE requests SET status = 'LOCKED' WHERE req_id = %s", (req_id,))
        conn.commit()
        
        print(f"Fetching comments for request {req_id}, video {video_id}")
        comments = fetch_comments(video_id, api_key, max_comments)
        
        if not comments:
            print(f"No comments found for video {video_id}")
            cur.execute("UPDATE requests SET status = 'FAILED' WHERE req_id = %s", (req_id,))
            conn.commit()
            return
        
        # Delete old comments for this video if they exist
        cur.execute("""
            DELETE FROM raw_comments 
            WHERE video_id = %s AND req_id IN (
                SELECT req_id FROM requests WHERE video_url = %s
            )
        """, (video_id, video_url))
        
        for c in comments:
            cur.execute("""
                INSERT INTO raw_comments (comment_id, video_id, req_id, text_display, scraped_at)
                VALUES (%s, %s, %s, %s, NOW())
                ON CONFLICT (comment_id) DO UPDATE
                SET text_display = EXCLUDED.text_display, scraped_at = NOW()
            """, (c['id'], video_id, req_id, c['text']))
        
        cur.execute("UPDATE requests SET status = 'FETCHED' WHERE req_id = %s", (req_id,))
        conn.commit()
        
        print(f"Saved {len(comments)} comments for request {req_id}")
        
    except Exception as e:
        print(f"Error: {e}")
        conn.rollback()
    finally:
        cur.close()
        conn.close()

if __name__ == "__main__":
    main()