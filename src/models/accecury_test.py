# accuracy_report_full.py - Fixed version for your CSV structure
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

class AccuracyTester:
    def __init__(self, model_path="../../models/3_class/myanmar_sentiment_1/"):
        print("Loading model...")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        # Force CPU to avoid CUDA issues
        self.device = torch.device("cpu")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Get labels from model config
        self.id2label = self.model.config.id2label
        if self.id2label is None:
            # Default for 3-class sentiment model
            self.id2label = {0: "negative", 1: "neutral", 2: "positive"}
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.model.to(self.device)
        self.model.eval()
        self.emotions = list(self.id2label.values())
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Emotions: {', '.join(self.emotions)}")
        print(f"📊 Number of classes: {len(self.emotions)}\n")
    
    def predict(self, text):
        """Predict sentiment for a single text"""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
        
        return self.id2label[pred], confidence
    
    def test_on_csv(self, csv_path="../../data/accuracy_test/test1_pro.csv"):
        """Test model on entire CSV file"""
        
        print(f"📂 Loading CSV from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        print(f"📊 Columns: {df.columns.tolist()}")
        
        # FIXED: Check for correct column names
        # Your CSV has 'cleaned_test' instead of 'cleaned_text'
        text_column = None
        if 'cleaned_text' in df.columns:
            text_column = 'cleaned_text'
        elif 'cleaned_test' in df.columns:
            text_column = 'cleaned_test'
            print("📝 Using 'cleaned_test' as text column")
        elif 'text' in df.columns:
            text_column = 'text'
        else:
            # Try to find any text-like column
            for col in df.columns:
                if df[col].dtype == 'object':
                    text_column = col
                    print(f"📝 Using '{col}' as text column")
                    break
        
        if text_column is None:
            raise ValueError("No text column found! Please check CSV columns.")
        
        # Standardize emotion labels
        label_column = None
        if 'emotion_class' in df.columns:
            label_column = 'emotion_class'
        elif 'emotion' in df.columns:
            label_column = 'emotion'
        elif 'label' in df.columns:
            label_column = 'label'
        
        if label_column is None:
            raise ValueError("No emotion column found! Please check CSV columns.")
        
        print(f"📝 Using '{label_column}' as label column")
        
        # Clean emotion labels - convert to lowercase for consistency
        df[label_column] = df[label_column].astype(str).str.strip().str.lower()
        
        # Show unique labels before mapping
        print(f"\n📊 Unique labels in CSV: {df[label_column].unique()}")
        
        # Map labels to model's expected format
        # Adjust this mapping based on your actual labels
        label_mapping = {
            'positive': 'positive',
            'pos': 'positive',
            'happy': 'positive',
            'joy': 'positive',
            'good': 'positive',
            'negative': 'negative',
            'neg': 'negative',
            'sad': 'negative',
            'anger': 'negative',
            'bad': 'negative',
            'neutral': 'neutral',
            'neu': 'neutral',
            'natural': 'neutral',
            'normal': 'neutral'
        }
        
        # Apply mapping
        df['mapped_emotion'] = df[label_column].map(label_mapping)
        
        # Show unmapped labels
        unmapped = df[df['mapped_emotion'].isna()]
        if len(unmapped) > 0:
            print(f"\n⚠️ Unmapped labels found: {unmapped[label_column].unique()}")
            print("These samples will be skipped.")
        
        # Remove unmapped rows
        df = df.dropna(subset=['mapped_emotion'])
        
        # Filter to only emotions that match the model
        df = df[df['mapped_emotion'].isin(self.emotions)]
        
        if len(df) == 0:
            print(f"\n❌ No valid samples after filtering!")
            print(f"Model expects: {self.emotions}")
            print(f"Available after mapping: {df['mapped_emotion'].unique()}")
            return None, None, df
        
        print(f"\n📊 After filtering: {len(df)} samples with valid emotions")
        print(f"📊 Label distribution:")
        for emotion in self.emotions:
            count = (df['mapped_emotion'] == emotion).sum()
            if count > 0:
                print(f"  {emotion}: {count} samples")
        
        print("\n🔄 Running predictions on entire dataset...")
        
        y_true = []
        y_pred = []
        confidences = []
        
        # Run predictions with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            text = row[text_column]
            true_label = row['mapped_emotion']
            
            # Handle NaN or empty text
            if pd.isna(text) or str(text).strip() == "":
                print(f"⚠️ Skipping empty text at index {_.name}")
                continue
            
            predicted, confidence = self.predict(str(text))
            
            y_true.append(true_label)
            y_pred.append(predicted)
            confidences.append(confidence)
        
        if len(y_true) == 0:
            print("❌ No valid predictions made!")
            return None, None, df
        
        # Add predictions to dataframe (only for rows that were processed)
        df_valid = df.iloc[:len(y_true)].copy()
        df_valid['predicted'] = y_pred
        df_valid['confidence'] = confidences
        df_valid['correct'] = df_valid['predicted'] == df_valid['mapped_emotion']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Print results
        print("\n" + "="*60)
        print("MYANMAR SENTIMENT CLASSIFICATION - FULL CSV TEST")
        print("="*60)
        
        print(f"\n📊 DATASET INFO")
        print("-"*50)
        print(f"  Total samples: {len(y_true)}")
        print(f"  Classes: {', '.join(self.emotions)}")
        
        print(f"\n📈 OVERALL PERFORMANCE")
        print("-"*50)
        print(f"  Accuracy:      {accuracy:.3f} ({accuracy:.1%})")
        print(f"  F1-Macro:      {f1_macro:.3f}")
        print(f"  F1-Weighted:   {f1_weighted:.3f}")
        print(f"  Avg Confidence: {np.mean(confidences):.3f} ({np.mean(confidences):.1%})")
        
        print(f"\n📊 PER-CLASS PERFORMANCE")
        print("-"*50)
        report = classification_report(y_true, y_pred, target_names=self.emotions, digits=3)
        print(report)
        
        print(f"\n📊 DETAILED BREAKDOWN BY SENTIMENT")
        print("-"*50)
        
        for emotion in self.emotions:
            emotion_data = df_valid[df_valid['mapped_emotion'] == emotion]
            if len(emotion_data) > 0:
                correct = emotion_data['correct'].sum()
                total = len(emotion_data)
                acc = correct / total
                avg_conf = emotion_data['confidence'].mean()
                print(f"  {emotion:10} : {int(correct)}/{total} = {acc:.1%}  (avg conf: {avg_conf:.1%})")
        
        # Confusion Matrix
        print(f"\n📊 CONFUSION MATRIX")
        print("-"*50)
        cm = confusion_matrix(y_true, y_pred, labels=self.emotions)
        
        # Print confusion matrix as table
        print(f"{'':12}", end="")
        for emotion in self.emotions:
            print(f"{emotion[:6]:<10}", end="")
        print()
        
        for i, emotion_true in enumerate(self.emotions):
            print(f"{emotion_true:<12}", end="")
            for j in range(len(self.emotions)):
                print(f"{cm[i][j]:<10}", end="")
            print()
        
        # Plot confusion matrix
        try:
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.emotions, 
                       yticklabels=self.emotions)
            plt.title('Confusion Matrix - Sentiment Classification')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig('confusion_matrix.png')
            print("\n✅ Confusion matrix saved to: confusion_matrix.png")
        except Exception as e:
            print(f"\n⚠️ Could not save confusion matrix: {e}")
        
        # Save results
        output_path = csv_path.replace('.csv', '_with_predictions.csv')
        df_valid.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Show sample of misclassifications
        print(f"\n📝 SAMPLE MISCLASSIFICATIONS (First 10)")
        print("-"*50)
        misclassified = df_valid[~df_valid['correct']]
        if len(misclassified) > 0:
            for _, row in misclassified.head(10).iterrows():
                text_preview = row[text_column][:60] if len(str(row[text_column])) > 60 else row[text_column]
                print(f"  Text: {text_preview}...")
                print(f"    True: {row['mapped_emotion']} → Predicted: {row['predicted']} (conf: {row['confidence']:.1%})")
        else:
            print("  No misclassifications found!")
        
        # Show correct predictions sample
        print(f"\n📝 SAMPLE CORRECT PREDICTIONS (First 10)")
        print("-"*50)
        correct = df_valid[df_valid['correct']]
        if len(correct) > 0:
            for _, row in correct.head(10).iterrows():
                text_preview = row[text_column][:60] if len(str(row[text_column])) > 60 else row[text_column]
                print(f"  Text: {text_preview}...")
                print(f"    ✅ {row['mapped_emotion']} (conf: {row['confidence']:.1%})")
        
        print("\n" + "="*60)
        if accuracy >= 0.85:
            print("🏆 FINAL VERDICT: EXCELLENT - PRODUCTION READY")
        elif accuracy >= 0.75:
            print("✅ FINAL VERDICT: GOOD - READY FOR USE")
        elif accuracy >= 0.65:
            print("⚠️ FINAL VERDICT: ACCEPTABLE - CAN IMPROVE")
        else:
            print("❌ FINAL VERDICT: NEEDS MORE WORK")
        print("="*60)
        
        return accuracy, f1_macro, df_valid
    
    def run_test(self, csv_path=None):
        """Run test on the CSV file"""
        if csv_path is None:
            csv_path = "../../data/accuracy_test/test1_pro.csv"
        return self.test_on_csv(csv_path)

if __name__ == "__main__":
    # Optional: specify custom paths
    # tester = AccuracyTester(model_path="/path/to/your/model")
    # acc, f1, df = tester.run_test(csv_path="/path/to/your/test.csv")
    
    tester = AccuracyTester()
    acc, f1, df = tester.run_test()