# model_dev.py - Complete Fixed Version
import pandas as pd
import torch
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# Force CPU usage (updated for newer transformers)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load your existing model
model_path = "../../models/trained/xlm-roberta-base-emotion"

print(f"Looking for model at: {os.path.abspath(model_path)}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model not found at {model_path}")

print("Loading your model...")
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Set device to CPU explicitly
device = torch.device("cpu")
model.to(device)
print(f"✅ Model loaded on: {device}")

# Create NEW training data with STANDARDIZED labels (all capitalized)
new_data = pd.DataFrame([
    # Neutral examples
    ("ဒီ နေ့ ရာသီ ဥတု က သာ မာန် ပဲ", "Neutral"),
    ("အပူ ချိန် က အ သင့် အ တင့် ရှိ တယ်", "Neutral"),
    ("လေ တိုက် နှုန်း က ပုံ မ မှန် ပဲ", "Neutral"),
    ("မိုး အ နည်း အ ငယ် ရွာ သွန်း တယ်", "Neutral"),
    ("ရုံး က အ စည်း အ ဝေး ပုံ မ မှန် ပါ ပဲ", "Neutral"),
    ("အိမ် အလုပ် တွေ လုပ် နေ တယ်", "Neutral"),
    ("ထမင်း စား ပြီး ပြီ", "Neutral"),
    ("ဖုန်း မြည် နေ တယ်", "Neutral"),
    ("ကွန်ပျူတာ သုံး နေ တယ်", "Neutral"),
    ("စာ ဖတ် နေ တယ်", "Neutral"),
    ("ရေ သောက် နေ တယ်", "Neutral"),
    ("အိပ် နေ တယ်", "Neutral"),
    
    # Anger examples
    ("ငါ အ ရမ်း ဒေါသ ကြီး နေ တယ်", "Anger"),
    ("ဒါ က ငါ့ ကို စိတ် တို စေ တယ်", "Anger"),
    ("မင်း ရဲ့ လုပ် ရပ် က ငါ့ ကို ဒေါသ ထွက် စေ တယ်", "Anger"),
    ("ဒီ အ ခြေ အ နေ က ငါ့ ကို အ ရမ်း စိတ် ဆိုး စေ တယ်", "Anger"),
    ("ငါ ဒေါသ နဲ့ ပေါက် ကွဲ လု မ ဆိုး အောင် နေ တယ်", "Anger"),
    ("ဒါ က လုံး ဝ မ ကြိုက် ဘူး", "Anger"),
    ("ငါ အ ရမ်း မကျေ နပ် ဘူး", "Anger"),
    ("ဘာ ကြောင့် ဒီ လို လုပ် တာ လဲ", "Anger"),
    ("ငါ စိတ် ရှုပ် နေ တယ်", "Anger"),
    ("ဒါ က သည်း ခံ လို့ မ ရ ဘူး", "Anger"),
    
    # Surprise examples
    ("အင်း ဒါ မျှော် လင့် ထား တာ မ ဟုတ် ဘူး", "Surprise"),
    ("ဒါ က ငါ့ ကို အံ့အား သင့် စေ တယ်", "Surprise"),
    ("ဒါ ဖြစ် မယ် လို့ မ ထင် ဘူး", "Surprise"),
    ("ဒါ က တကယ် မ မျှော် လင့် ထား ဘူး", "Surprise"),
    ("ဒီ အ ကြောင်း ကို ကြား ရ တာ အံ့သြ စရာ ပဲ", "Surprise"),
    ("အင်း ဒါ မှန်လား", "Surprise"),
    ("ဘယ် လို ဖြစ် နိုင် မလဲ", "Surprise"),
    ("ဒါ မှန်း မ သိ ဘူး", "Surprise"),
    ("အံ့သြ စရာ ကောင်း လိုက် တာ", "Surprise"),
], columns=['cleaned_text', 'emotion_class'])

# Load original data and STANDARDIZE labels
df_original = pd.read_csv("../../data/processed/dataset_processed.csv")

# Standardize original labels (capitalize first letter)
df_original['emotion_class'] = df_original['emotion_class'].str.capitalize()

# Fix common spelling issues
df_original['emotion_class'] = df_original['emotion_class'].replace('Suprise', 'Surprise')

# Remove any rows with NaN or invalid labels
df_original = df_original[df_original['emotion_class'].isin(['Anger', 'Fear', 'Joy', 'Love', 'Neutral', 'Sadness', 'Surprise'])]

# Sample from original
original_sample = df_original.sample(n=200, random_state=42)

# Combine datasets
df_train = pd.concat([original_sample, new_data], ignore_index=True)

print(f"\n📊 Training on {len(df_train)} samples:")
print(f"  - Original: {len(original_sample)}")
print(f"  - New: {len(new_data)}")
print(f"\nData distribution after combining:")
print(df_train['emotion_class'].value_counts())

# Get unique labels (should be 7 standard emotions)
unique_labels = sorted(df_train['emotion_class'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
df_train['label'] = df_train['emotion_class'].map(label2id)

print(f"\n✅ Standardized labels: {label2id}")

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_train['cleaned_text'].tolist(),
    df_train['label'].tolist(),
    test_size=0.1,
    random_state=42,
    stratify=df_train['label']
)

# Tokenize
def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=64,
        return_tensors=None
    )

print("\n🔄 Tokenizing...")
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

train_dataset = Dataset.from_dict({
    'input_ids': train_encodings['input_ids'],
    'attention_mask': train_encodings['attention_mask'],
    'labels': train_labels
})

val_dataset = Dataset.from_dict({
    'input_ids': val_encodings['input_ids'],
    'attention_mask': val_encodings['attention_mask'],
    'labels': val_labels
})

# Training arguments (removed 'no_cuda' - use 'use_cpu' instead)
training_args = TrainingArguments(
    output_dir="./fine_tuned",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-5,
    warmup_steps=10,
    eval_strategy="epoch",
    save_strategy="no",
    logging_steps=5,
    report_to="none",
    use_cpu=True,  # Changed from 'no_cuda' to 'use_cpu'
)

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='weighted')
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

print("\n🚀 Fine-tuning your model (CPU mode)...")
print("⏱️  This will take 10-15 minutes...")

# Train
trainer.train()

# Evaluate
print("\n📊 Evaluation results:")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.3f}")
print(f"F1-Score: {eval_results['eval_f1']:.3f}")

# Save improved model
improved_model_path = "models/trained/xlm-roberta-improved"
model.save_pretrained(improved_model_path)
tokenizer.save_pretrained(improved_model_path)

print(f"\n✅ Model improved and saved to: {improved_model_path}")

# Quick test
print("\n🎯 Testing improved model:")
test_texts = [
    ("ဒီ နေ့ ရာသီ ဥတု က သာ မာန် ပဲ", "Neutral"),
    ("ငါ အ ရမ်း ဒေါသ ထွက် တယ်", "Anger"),
    ("အံ့သြ စရာ ပဲ", "Surprise"),
    ("ငါ မင်း ကို ချစ် တယ်", "Love"),
]

model.eval()
for text, expected in test_texts:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
        predicted = id2label[pred]
        confidence = torch.max(probs).item()
    
    status = "✅" if predicted == expected else "❌"
    print(f"\n{status} Expected: {expected}")
    print(f"   Text: {text[:40]}...")
    print(f"   Predicted: {predicted} ({confidence:.1%})")

print("\n✅ Fine-tuning complete!")