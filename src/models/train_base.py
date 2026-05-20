import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
from sklearn.utils.class_weight import compute_class_weight

# Load your data
df = pd.read_csv("dataset_processed.csv")  # Change to your file name
print(f"Dataset shape: {df.shape}")
print("\nEmotion distribution:")
print(df['emotion_class'].value_counts())

# Clean and standardize labels
df['emotion_class'] = df['emotion_class'].astype(str).str.strip()
df['emotion_class'] = df['emotion_class'].replace('suprise', 'Surprise')
df['emotion_class'] = df['emotion_class'].str.capitalize()
df = df[~df['emotion_class'].str.lower().isin(['nan', 'none', ''])]

# Get unique labels
unique_labels = sorted(df['emotion_class'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

print(f"\nLabels: {label2id}")
print(f"Number of classes: {num_labels}")

# Add label column
df['label'] = df['emotion_class'].map(label2id)

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned_text'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Compute class weights for imbalance
class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
weights = torch.tensor(class_weights, dtype=torch.float)
if torch.cuda.is_available():
    weights = weights.cuda()

print(f"\nClass weights: {dict(zip(unique_labels, class_weights))}")

# Load tokenizer and model
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(texts):
    return tokenizer(
        texts, 
        padding="max_length", 
        truncation=True, 
        max_length=64,
        return_tensors=None
    )

# Tokenize
train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Create datasets
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

# Load model
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True
)

# FIXED: Custom Trainer with weighted loss (compatible with new transformers version)
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove num_items_in_batch if we don't use it
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    return {'accuracy': accuracy, 'f1_score': f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./xlmr_myanmar_weighted",
    num_train_epochs=12,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    weight_decay=0.01,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1_score",
    greater_is_better=True,
    learning_rate=3e-5,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none",
    gradient_accumulation_steps=2,
)

# Use WeightedTrainer
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("\n" + "=" * 50)
print("STARTING TRAINING WITH CLASS WEIGHTS")
print("=" * 50)
trainer.train()

# Evaluate
print("\n" + "=" * 50)
print("EVALUATION RESULTS")
print("=" * 50)
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.3f}")
print(f"F1-Score: {eval_results['eval_f1_score']:.3f}")

# Detailed classification report
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
print("\nDetailed Classification Report:")
print(classification_report(val_labels, y_pred, target_names=unique_labels))

# Save model
model_save_path = "./myanmar_emotion_weighted_final"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save label mapping
with open(f"{model_save_path}/label_mapping.json", "w", encoding="utf-8") as f:
    json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved to {model_save_path}")

# Test function
def predict_emotion(text, model, tokenizer, id2label):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        predictions = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(predictions, dim=1).item()
        confidence = torch.max(predictions).item()
    
    return id2label[predicted_class], confidence

# Test examples
print("\n" + "=" * 50)
print("TESTING ON SAMPLE TEXTS")
print("=" * 50)

test_examples = [
    "ပန်း ခြံ ထဲ မှာ လှ ပ တဲ့ နေ့ လေး တစ် နေ့ ကို ပျော် ရွှင် စွာ ဖြတ် သန်း နေ ပါ တယ်",
    "ဒီ မ နက် လမ်း အ ရမ်း ပိတ် တယ်",
    "ငါ မင်း ကို အရမ်း ချစ် တယ်",
    "ငါ အရမ်း ကြောက် နေ တယ်",
    "ဒီ နေ့ အ လုပ် က ပုံ မ မှန် ပဲ",
]

for text in test_examples:
    emotion, confidence = predict_emotion(text, model, tokenizer, id2label)
    print(f"\nText: {text[:50]}...")
    print(f"Predicted: {emotion} (Confidence: {confidence:.2%})")