import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ===== LOAD YOUR DATA =====
df = pd.read_csv("../../data/processed/dataset_processed.csv")

texts = df["cleaned_text"].tolist()
true_labels = df["emotion_class"].tolist()

# ===== LOAD YOUR MODEL =====
# Example: HuggingFace pipeline
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="../../models/trained/xlm-roberta-improved-91",
    tokenizer="../../models/trained/xlm-roberta-improved-91",
    return_all_scores=False
)

# ===== PREDICT =====
pred_labels = []

for text in texts:
    result = classifier(text)[0]
    pred_labels.append(result["label"])

# ===== EVALUATION =====
acc = accuracy_score(true_labels, pred_labels)

print("\n🎯 Accuracy:", round(acc * 100, 2), "%")

print("\n📊 Classification Report:")
print(classification_report(true_labels, pred_labels))

print("\n🧩 Confusion Matrix:")
print(confusion_matrix(true_labels, pred_labels))