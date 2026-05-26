#!/usr/bin/env python3
# test_bot.py - Place in project root: ~/Desktop/myanmar_sentiment/test_bot.py

import os
import sys
import yaml
import torch
import asyncio
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Load config
with open('configs/config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Load model
model_path = config['pipeline']['model_path']
if model_path.startswith('.'):
    model_path = os.path.join(os.path.dirname(__file__), model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

label_map = {0: 'Anger', 1: 'Fear', 2: 'Joy', 3: 'postitive', 4: 'Neutral', 5: 'Sadness', 6: 'Surprise'}

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128, padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_id].item()
    
    return label_map[pred_id], confidence

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "🇲🇲 Myanmar Sentiment Bot\n\n"
        "Send me any text in Myanmar (Burmese) and I'll detect the emotion.\n\n"
        "Emotions: Anger, Fear, Joy, postitive, Neutral, Sadness, Surprise"
    )

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    await update.message.chat.send_action(action="typing")
    
    try:
        emotion, confidence = predict_sentiment(text)
        response = f"🎭 Emotion: {emotion}\n📊 Confidence: {confidence:.2%}"
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Error: {e}")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Send me any text and I'll analyze the emotion.")

def main():
    token = config['telegram']['bot_token']
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    print("🤖 Test bot running... Press Ctrl+C to stop")
    print(f"Model loaded from: {model_path}")
    print(f"Using device: {device}")
    app.run_polling()

if __name__ == "__main__":
    main()