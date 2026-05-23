import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import LabelEncoder
from cleanlab.filter import find_label_issues
import warnings
warnings.filterwarnings('ignore')

# Get the directory where THIS script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'combined.csv')

print(f"📁 Loading CSV from: {csv_path}")

# Load your data
df = pd.read_csv(csv_path)

# Clean emotion classes (fix 'suprise' typo)
df['emotion_class'] = df['emotion_class'].replace('suprise', 'surprise')

# Remove rows with missing labels if any
df = df.dropna(subset=['emotion_class'])

print(f"✅ Loaded {len(df)} rows")
print(f"📊 Unique labels: {df['emotion_class'].unique()}")

# Prepare features and labels
X_text = df['text'].astype(str)
y_str = df['emotion_class']

# Encode string labels to integers (Cleanlab requires integers)
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_str)

# Map back for display
label_mapping = dict(zip(label_encoder.transform(label_encoder.classes_), label_encoder.classes_))
print(f"📋 Label mapping: {label_mapping}")

# Convert text to numerical features
print("🔄 Converting text to features...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X = vectorizer.fit_transform(X_text)

# Get out-of-sample predicted probabilities
print("🔄 Training model and getting predictions...")
model = LogisticRegression(max_iter=1000)
pred_probs = cross_val_predict(model, X, y, cv=3, method='predict_proba')

# Find label issues
print("🔍 Finding label issues...")
label_issues = find_label_issues(
    labels=y,
    pred_probs=pred_probs,
    return_indices_ranked_by='self_confidence'
)

print(f"\n{'='*60}")
print(f"🔴 Found {len(label_issues)} potential labeling errors")
print(f"{'='*60}\n")

# Display problematic rows
for i, idx in enumerate(label_issues[:30]):  # Show top 30
    print(f"\n📌 Issue #{i+1} (Original row {idx + 2}):")
    print(f"   Text: {X_text.iloc[idx][:120]}...")
    original_label = y_str.iloc[idx]
    print(f"   ❌ Your label: {original_label}")
    
    # Show top 3 predicted classes
    proba = pred_probs[idx]
    classes = label_encoder.classes_
    top3_idx = np.argsort(proba)[-3:][::-1]
    suggested_label = classes[top3_idx[0]]
    confidence = proba[top3_idx[0]]
    print(f"   ✅ Model suggests: {suggested_label} ({confidence:.2f})")
    print(f"   Other possibilities: {classes[top3_idx[1]]} ({proba[top3_idx[1]]:.2f})")
    if len(classes) >= 3:
        print(f"                       {classes[top3_idx[2]]} ({proba[top3_idx[2]]:.2f})")
    print("-" * 60)

# Save results to CSV
results_data = []
for idx in label_issues:
    proba = pred_probs[idx]
    classes = label_encoder.classes_
    suggested_idx = np.argmax(proba)
    results_data.append({
        'row_number': idx + 2,
        'text': X_text.iloc[idx],
        'current_label': y_str.iloc[idx],
        'suggested_label': classes[suggested_idx],
        'confidence': proba[suggested_idx],
        'second_best_label': classes[np.argsort(proba)[-2]] if len(classes) > 1 else '',
        'second_best_confidence': proba[np.argsort(proba)[-2]] if len(classes) > 1 else 0
    })

results_df = pd.DataFrame(results_data)
output_path = os.path.join(script_dir, 'label_issues_report.csv')
results_df.to_csv(output_path, index=False)
print(f"\n📄 Full report saved to: {output_path}")

# Summary statistics
print(f"\n📊 Summary of potential labeling errors by current label:")
summary = results_df['current_label'].value_counts()
for label, count in summary.items():
    print(f"   {label}: {count} potential errors")

print(f"\n📊 Top suggested corrections:")
for label, count in results_df['suggested_label'].value_counts().head(10).items():
    print(f"   Change to {label}: {count} times")