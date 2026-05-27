import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch
import json
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# Load your data
df = pd.read_csv("combined_processed.csv")  # Change to your file name
print(f"Dataset shape: {df.shape}")
print("\nSentiment distribution (before filtering):")
print(df['emotion_class'].value_counts())

# Clean and standardize labels
df['emotion_class'] = df['emotion_class'].astype(str).str.strip()
df = df[~df['emotion_class'].str.lower().isin(['nan', 'none', ''])]

# Filter to keep only positive, negative, neutral (3 classes)
desired_classes = ['positive', 'negative', 'neutral']
df = df[df['emotion_class'].isin(desired_classes)]

# Reset index after filtering
df = df.reset_index(drop=True)

# Get unique labels (should be 3)
unique_labels = sorted(df['emotion_class'].unique())
label2id = {label: i for i, label in enumerate(unique_labels)}
id2label = {i: label for label, i in label2id.items()}
num_labels = len(unique_labels)

print(f"\nLabels: {label2id}")
print(f"Number of classes: {num_labels}")

# Verify we have exactly 3 classes
if num_labels != 3:
    print(f"WARNING: Found {num_labels} classes instead of 3!")
    print("Available classes:", unique_labels)

# Add label column
df['label'] = df['emotion_class'].map(label2id)

# Train/validation split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(),  # Changed from 'cleaned_text' to 'text' as per your CSV
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"\nTraining samples: {len(train_texts)}")
print(f"Validation samples: {len(val_texts)}")

# Print class distribution in training set
print("\nTraining set distribution:")
for label, count in pd.Series(train_labels).value_counts().items():
    print(f"  {id2label[label]}: {count} samples")

# Custom class weights with extra weight on neutral class
# First compute standard balanced weights
# Manual weights (direct assignment)
class_weight_dict = {
    'positive': 0.89,   # Adjust as needed
    'negative': 0.89,   # Higher = more importance
    'neutral': 1.33    # Highest importance
}

# Convert to tensor
weights = torch.tensor([class_weight_dict[label] for label in unique_labels], dtype=torch.float)
print(f"Custom weights (neutral weighted 3x more): {class_weight_dict}")

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

# Custom Trainer with weighted loss
# Custom Trainer with weighted loss - FIXED VERSION
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get('logits')
        
        # Move weights to the same device as logits (GPU)
        device = logits.device
        weights_on_device = weights.to(device)
        
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights_on_device)
        loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average='weighted')
    # Also compute per-class metrics
    per_class_f1 = f1_score(labels, predictions, average=None)
    return {
        'accuracy': accuracy, 
        'f1_score': f1,
        'f1_positive': per_class_f1[0] if len(per_class_f1) > 0 else 0,
        'f1_negative': per_class_f1[1] if len(per_class_f1) > 1 else 0,
        'f1_neutral': per_class_f1[2] if len(per_class_f1) > 2 else 0
    }

# Training arguments
training_args = TrainingArguments(
    output_dir="./xlmr_sentiment_weighted",
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
print("STARTING TRAINING WITH CUSTOM CLASS WEIGHTS (Neutral gets 3x weight)")
print("=" * 50)
trainer.train()

# Evaluate on validation set
print("\n" + "=" * 50)
print("EVALUATION RESULTS ON VALIDATION SET")
print("=" * 50)
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.3f}")
print(f"F1-Score (Weighted): {eval_results['eval_f1_score']:.3f}")
if 'eval_f1_positive' in eval_results:
    print(f"F1-Score (Positive): {eval_results['eval_f1_positive']:.3f}")
    print(f"F1-Score (Negative): {eval_results['eval_f1_negative']:.3f}")
    print(f"F1-Score (Neutral): {eval_results['eval_f1_neutral']:.3f}")

# Detailed classification report for validation set
predictions = trainer.predict(val_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
print("\nDetailed Classification Report (Validation Set):")
print(classification_report(val_labels, y_pred, target_names=unique_labels))

# Confusion Matrix for validation set
cm = confusion_matrix(val_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=unique_labels, 
            yticklabels=unique_labels)
plt.title('Confusion Matrix - Validation Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# ===== TEST ON ANOTHER CSV FILE =====
print("\n" + "=" * 50)
print("TESTING ON ANOTHER CSV FILE")
print("=" * 50)

def test_on_csv(csv_path, model, tokenizer, label2id, id2label, unique_labels):
    """Test the model on a CSV file and return metrics"""
    try:
        test_df = pd.read_csv(csv_path)
        print(f"\nTesting on: {csv_path}")
        print(f"Test CSV shape: {test_df.shape}")
        
        # Check available columns
        print(f"Columns found: {test_df.columns.tolist()}")
        
        # Determine which column to use for text
        text_column = None
        for col in ['text', 'cleaned_text', 'sentence', 'content']:
            if col in test_df.columns:
                text_column = col
                break
        
        label_column = 'emotion_class'
        
        if text_column is None:
            print(f"No text column found in {csv_path}")
            return None
        
        if label_column not in test_df.columns:
            print(f"No '{label_column}' column found in {csv_path}")
            return None
        
        # Clean test data
        test_df[label_column] = test_df[label_column].astype(str).str.strip()
        test_df = test_df[~test_df[label_column].str.lower().isin(['nan', 'none', ''])]
        
        # Filter to only the 3 classes
        test_df = test_df[test_df[label_column].isin(unique_labels)]
        
        if len(test_df) == 0:
            print(f"No valid test samples found for classes {unique_labels}")
            return None
        
        # Map labels
        test_df['label'] = test_df[label_column].map(label2id)
        
        # Remove rows with unmapped labels
        test_df = test_df.dropna(subset=['label'])
        test_df['label'] = test_df['label'].astype(int)
        
        # Prepare test texts and labels
        test_texts = test_df[text_column].tolist()
        test_labels = test_df['label'].tolist()
        
        # Tokenize test data
        test_encodings = tokenize_function(test_texts)
        
        # Create test dataset
        test_dataset = Dataset.from_dict({
            'input_ids': test_encodings['input_ids'],
            'attention_mask': test_encodings['attention_mask'],
            'labels': test_labels
        })
        
        # Predict on test set
        test_predictions = trainer.predict(test_dataset)
        y_pred_test = np.argmax(test_predictions.predictions, axis=1)
        
        # Calculate test metrics
        test_accuracy = accuracy_score(test_labels, y_pred_test)
        test_f1 = f1_score(test_labels, y_pred_test, average='weighted')
        per_class_f1 = f1_score(test_labels, y_pred_test, average=None)
        
        print(f"\n📊 Test Results on {csv_path}:")
        print(f"Test samples: {len(test_texts)}")
        print(f"Accuracy: {test_accuracy:.3f}")
        print(f"Weighted F1-Score: {test_f1:.3f}")
        print(f"\nPer-class F1-Scores:")
        for i, label in enumerate(unique_labels):
            print(f"  {label}: {per_class_f1[i]:.3f}")
        
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, y_pred_test, target_names=unique_labels))
        
        # Confusion Matrix for test set
        cm_test = confusion_matrix(test_labels, y_pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=unique_labels, 
                   yticklabels=unique_labels)
        plt.title(f'Confusion Matrix - {csv_path}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
        return {
            'accuracy': test_accuracy,
            'f1_score': test_f1,
            'per_class_f1': dict(zip(unique_labels, per_class_f1)),
            'predictions': y_pred_test,
            'true_labels': test_labels
        }
        
    except FileNotFoundError:
        print(f"❌ File not found: {csv_path}")
        return None
    except Exception as e:
        print(f"❌ Error testing {csv_path}: {e}")
        return None

# Test on multiple CSV files
test_csv_files = [
    "test_data.csv",      # Change to your test file path
    "validation.csv",     # Add more files as needed
    "evaluation.csv"
]

results_summary = {}
for csv_file in test_csv_files:
    result = test_on_csv(csv_file, model, tokenizer, label2id, id2label, unique_labels)
    if result:
        results_summary[csv_file] = result

# Print summary of all test results
if results_summary:
    print("\n" + "=" * 50)
    print("SUMMARY OF ALL TEST RESULTS")
    print("=" * 50)
    summary_df = pd.DataFrame([
        {
            'File': file,
            'Accuracy': res['accuracy'],
            'F1-Score': res['f1_score'],
            'Positive F1': res['per_class_f1'].get('positive', 0),
            'Negative F1': res['per_class_f1'].get('negative', 0),
            'Neutral F1': res['per_class_f1'].get('neutral', 0)
        }
        for file, res in results_summary.items()
    ])
    print(summary_df.to_string(index=False))

# Save model
model_save_path = "./myanmar_sentiment_neutral_weighted"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# Save label mapping and class weights info
metadata = {
    "label2id": label2id,
    "id2label": id2label,
    "class_weights": class_weight_dict,
    "num_classes": num_labels,
    "model_type": "xlm-roberta-base",
    "neutral_weight_multiplier": 3.0
}

with open(f"{model_save_path}/metadata.json", "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"\n✅ Model saved to {model_save_path}")

# Test function with sentiment probabilities
def predict_sentiment(text, model, tokenizer, id2label):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = torch.max(probabilities).item()
        all_probabilities = probabilities[0].cpu().numpy()
    
    return id2label[predicted_class], confidence, all_probabilities

# Test examples with Burmese text
print("\n" + "=" * 50)
print("TESTING ON SAMPLE TEXTS")
print("=" * 50)

test_examples = [
    "ဒီနေ့ အရမ်းပျော်တယ်",  # Happy
    "ငါ အရမ်းစိတ်ဆိုးတယ်",  # Angry/Negative
    "ဒါကပုံမှန်ပါပဲ",  # Neutral
    "ကျေးဇူးတင်ပါတယ်",  # Thank you (positive)
    "အဆင်ပြေပါတယ်",  # It's okay (neutral)
    "မင်းကို မုန်းတယ်",  # I hate you (negative)
]

for text in test_examples:
    sentiment, confidence, probs = predict_sentiment(text, model, tokenizer, id2label)
    print(f"\nText: {text}")
    print(f"Predicted: {sentiment.upper()} (Confidence: {confidence:.2%})")
    print("Sentiment probabilities:")
    for i, prob in enumerate(probs):
        print(f"  {id2label[i]}: {prob:.2%}")

# Function to test on a batch from CSV
def evaluate_on_csv_batch(csv_path, model, tokenizer, label2id, id2label, num_samples=None):
    """Quick evaluation on a CSV file"""
    try:
        df_test = pd.read_csv(csv_path)
        
        # Find text column
        text_col = 'text' if 'text' in df_test.columns else 'cleaned_text'
        
        if num_samples:
            df_test = df_test.head(num_samples)
        
        results = []
        for idx, row in df_test.iterrows():
            text = row[text_col]
            true_label = row['emotion_class']
            pred_label, confidence, _ = predict_sentiment(text, model, tokenizer, id2label)
            results.append({
                'text': text[:50],
                'true_label': true_label,
                'predicted_label': pred_label,
                'confidence': confidence,
                'correct': true_label == pred_label
            })
        
        results_df = pd.DataFrame(results)
        accuracy = results_df['correct'].mean()
        
        print(f"\n📊 Batch Evaluation on {csv_path} (first {len(results_df)} samples):")
        print(f"Accuracy: {accuracy:.2%}")
        print("\nSample predictions:")
        print(results_df.head(10).to_string(index=False))
        
        return results_df
        
    except Exception as e:
        print(f"Error in batch evaluation: {e}")
        return None

# Run batch evaluation on test file
evaluate_on_csv_batch("test1.csv", model, tokenizer, label2id, id2label, num_samples=20)