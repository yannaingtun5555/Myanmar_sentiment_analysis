#!/usr/bin/env python3
"""
telegram_request_bot_fixed.py
"""

import os
import re
import logging
from datetime import datetime
from typing import Optional, Tuple, Dict, List
from urllib.parse import urlparse, parse_qs

import yaml
import psycopg2
from psycopg2.extras import RealDictCursor
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    filters,
    ContextTypes
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YouTubeRequestBot:
    def __init__(self, config_path: str = None):
        if not config_path:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            config_path = os.path.join(project_root, 'configs', 'config.yml')
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        db = self.config['database']
        self.db_params = {
            'host': db['host'],
            'port': db['port'],
            'database': db['database'],
            'user': db['user'],
            'password': db['password']
        }
        
        self.token = self.config['telegram']['bot_token']
        
        self.youtube_patterns = [
            r'(https?://)?(www\.)?(youtube\.com/watch\?v=)',
            r'(https?://)?(www\.)?(youtu\.be/)',
            r'(https?://)?(www\.)?(youtube\.com/shorts/)'
        ]
        
        logger.info("Bot initialized")
    
    def get_db_connection(self):
        return psycopg2.connect(**self.db_params)
    
    def extract_video_id(self, url: str) -> Optional[str]:
        parsed = urlparse(url)
        if 'youtu.be' in parsed.netloc:
            return parsed.path.lstrip('/')
        if 'youtube.com' in parsed.netloc:
            if 'shorts' in parsed.path:
                return parsed.path.split('/')[-1]
            query_params = parse_qs(parsed.query)
            return query_params.get('v', [None])[0]
        return None
    
    def is_youtube_url(self, text: str) -> bool:
        return any(re.search(pattern, text) for pattern in self.youtube_patterns)
    
    def create_request(self, user_id: int, video_url: str) -> Tuple[Optional[int], Optional[str]]:
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT req_id, status 
                        FROM requests 
                        WHERE user_id = %s AND video_url = %s 
                        AND status IN ('NEW', 'LOCKED', 'FETCHED')
                        LIMIT 1
                    """, (user_id, video_url))
                    
                    existing = cur.fetchone()
                    if existing:
                        return existing[0], f"exists|{existing[1]}"
                    
                    cur.execute("""
                        INSERT INTO requests (user_id, video_url, status)
                        VALUES (%s, %s, 'NEW')
                        RETURNING req_id
                    """, (user_id, video_url))
                    
                    req_id = cur.fetchone()[0]
                    conn.commit()
                    logger.info(f"Created request {req_id} for user {user_id}")
                    return req_id, None
        except Exception as e:
            logger.error(f"Create request error: {e}")
            return None, str(e)
    
    def get_user_requests(self, user_id: int, limit: int = 5) -> List[Dict]:
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT req_id, video_url, status, created_at
                        FROM requests 
                        WHERE user_id = %s 
                        ORDER BY created_at DESC 
                        LIMIT %s
                    """, (user_id, limit))
                    return cur.fetchall()
        except Exception as e:
            logger.error(f"Get user requests error: {e}")
            return []
    
    def get_request_status(self, req_id: int) -> Optional[Dict]:
        try:
            with self.get_db_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("""
                        SELECT r.req_id, r.video_url, r.status, r.created_at,
                               COUNT(rc.comment_id) as comment_count
                        FROM requests r
                        LEFT JOIN raw_comments rc ON r.req_id = rc.req_id
                        WHERE r.req_id = %s
                        GROUP BY r.req_id, r.video_url, r.status, r.created_at
                    """, (req_id,))
                    return cur.fetchone()
        except Exception as e:
            logger.error(f"Get request status error: {e}")
            return None

bot = YouTubeRequestBot()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    welcome_msg = (
        f"🇲🇲 Hello {user.first_name}!\n\n"
        "Send me a YouTube link to analyze comments.\n\n"
        "Commands:\n"
        "/start - Start\n"
        "/help - Help\n"
        "/status - Your requests"
    )
    
    keyboard = [
        [InlineKeyboardButton("Help", callback_data='help')],
        [InlineKeyboardButton("My Requests", callback_data='my_requests')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(welcome_msg, reply_markup=reply_markup)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    help_msg = (
        "How to use:\n"
        "1. Send a YouTube video URL\n"
        "2. Bot saves your request\n"
        "3. System analyzes comments\n"
        "4. Check status with /status\n\n"
        "Supported formats:\n"
        "- youtube.com/watch?v=VIDEO_ID\n"
        "- youtu.be/VIDEO_ID\n"
        "- youtube.com/shorts/VIDEO_ID"
    )
    await update.message.reply_text(help_msg)

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    
    if context.args:
        try:
            req_id = int(context.args[0])
            data = bot.get_request_status(req_id)
            if data:
                status_emoji = {
                    'NEW': '🟡', 
                    'LOCKED': '🔒', 
                    'FETCHED': '✅', 
                    'PREPROCESSED': '🔧',
                    'PREDICTED_MODEL': '🤖',
                    'FINALIZED': '📊',
                    'RESPONDED': '📨',
                    'FAILED': '❌'
                }.get(data['status'], '⚪')
                msg = f"{status_emoji} Request #{data['req_id']}\n"
                msg += f"Status: {data['status']}\n"
                msg += f"Comments: {data['comment_count']}\n"
                await update.message.reply_text(msg)
            else:
                await update.message.reply_text(f"Request #{req_id} not found")
        except ValueError:
            await update.message.reply_text("Use: /status 123")
        return
    
    requests = bot.get_user_requests(user_id)
    if requests:
        msg = "Your requests:\n\n"
        for req in requests:
            emoji = {
                'NEW': '🟡', 
                'LOCKED': '🔒', 
                'FETCHED': '✅', 
                'PREPROCESSED': '🔧',
                'PREDICTED_MODEL': '🤖',
                'FINALIZED': '📊',
                'RESPONDED': '📨',
                'FAILED': '❌'
            }.get(req['status'], '⚪')
            msg += f"{emoji} #{req['req_id']} - {req['status']}\n"
            msg += f"   {req['created_at'].strftime('%Y-%m-%d %H:%M')}\n\n"
        await update.message.reply_text(msg)
    else:
        await update.message.reply_text("No requests. Send me a YouTube link!")

async def handle_youtube_url(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    url = update.message.text.strip()
    
    if not bot.is_youtube_url(url):
        await update.message.reply_text("Invalid YouTube URL. Send a valid link.")
        return
    
    video_id = bot.extract_video_id(url)
    await update.message.reply_text(f"Processing video: {video_id}")
    
    req_id, error = bot.create_request(user.id, url)
    
    if error and error.startswith("exists|"):
        status = error.split("|")[1]
        await update.message.reply_text(f"Video already in queue. Request #{req_id} - Status: {status}")
    elif req_id:
        keyboard = [
            [InlineKeyboardButton("Check Status", callback_data=f'status_{req_id}')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(
            f"Request #{req_id} created! I'll analyze the comments.",
            reply_markup=reply_markup
        )
        logger.info(f"New request {req_id} from user {user.id} for video {video_id}")
    else:
        await update.message.reply_text("Error creating request. Try again later.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    if bot.is_youtube_url(text):
        await handle_youtube_url(update, context)
    else:
        await update.message.reply_text("Send me a YouTube video link")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    
    data = query.data
    user_id = update.effective_user.id
    
    if data == 'help':
        await query.edit_message_text(
            "Send any YouTube URL. I'll analyze the comments for sentiment."
        )
    elif data == 'my_requests':
        requests = bot.get_user_requests(user_id)
        if requests:
            msg = "Your requests:\n\n"
            for req in requests[:5]:
                emoji = {
                    'NEW': '🟡', 
                    'LOCKED': '🔒', 
                    'FETCHED': '✅', 
                    'PREPROCESSED': '🔧',
                    'PREDICTED_MODEL': '🤖',
                    'FINALIZED': '📊',
                    'RESPONDED': '📨',
                    'FAILED': '❌'
                }.get(req['status'], '⚪')
                msg += f"{emoji} #{req['req_id']} - {req['status']}\n"
            await query.edit_message_text(msg)
        else:
            await query.edit_message_text("No requests found")
    elif data.startswith('status_'):
        req_id = int(data.split('_')[1])
        req_data = bot.get_request_status(req_id)
        if req_data:
            emoji = {
                'NEW': '🟡', 
                'LOCKED': '🔒', 
                'FETCHED': '✅', 
                'PREPROCESSED': '🔧',
                'PREDICTED_MODEL': '🤖',
                'FINALIZED': '📊',
                'RESPONDED': '📨',
                'FAILED': '❌'
            }.get(req_data['status'], '⚪')
            msg = f"{emoji} Request #{req_data['req_id']}\n"
            msg += f"Status: {req_data['status']}\n"
            msg += f"Comments: {req_data['comment_count']}"
            await query.edit_message_text(msg)
        else:
            await query.edit_message_text(f"Request #{req_id} not found")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.error(f"Error: {context.error}")
    if update and update.effective_message:
        await update.effective_message.reply_text("An error occurred. Please try again.")

def main():
    print("\n" + "="*50)
    print("YouTube Sentiment Bot")
    print("="*50)
    
    application = Application.builder().token(bot.token).build()
    
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(CallbackQueryHandler(button_callback))
    application.add_error_handler(error_handler)
    
    print("Bot running... Press Ctrl+C to stop\n")
    application.run_polling()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nBot stopped")