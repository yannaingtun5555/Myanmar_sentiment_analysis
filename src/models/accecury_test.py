# accuracy_report_full.py - Test on entire CSV file
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from tqdm import tqdm

class AccuracyTester:
    def __init__(self, model_path="../../models/trained/xlm-roberta-improve-final-75"):
        print("Loading model...")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Get labels from model config
        self.id2label = self.model.config.id2label
        if self.id2label is None:
            self.id2label = {0: "Anger", 1: "Fear", 2: "Joy", 3: "Love", 4: "Neutral", 5: "Sadness", 6: "Surprise"}
        
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.emotions = list(self.id2label.values())
        
        print(f"✅ Model loaded on {self.device}")
        print(f"📊 Emotions: {', '.join(self.emotions)}\n")
    
    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
        
        return self.id2label[pred], confidence
    
    def test_on_csv(self, csv_path="../../data/processed/dataset_processed.csv"):
        """Test model on entire CSV file"""
        
        print(f"📂 Loading CSV from: {csv_path}")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")
        
        df = pd.read_csv(csv_path)
        print(f"📊 Loaded {len(df)} samples")
        
        # Standardize column names
        if 'cleaned_text' not in df.columns:
            if 'text' in df.columns:
                df.rename(columns={'text': 'cleaned_text'}, inplace=True)
        
        # Clean emotion labels
        if 'emotion_class' in df.columns:
            df['emotion_class'] = df['emotion_class'].str.capitalize()
        elif 'emotion_clean' in df.columns:
            df.rename(columns={'emotion_clean': 'emotion_class'}, inplace=True)
            df['emotion_class'] = df['emotion_class'].str.capitalize()
        
        # Fix common spelling
        df['emotion_class'] = df['emotion_class'].replace('Natural', 'Neutral')
        df['emotion_class'] = df['emotion_class'].replace('Suprise', 'Surprise')
        
        # Filter to only valid emotions
        df = df[df['emotion_class'].isin(self.emotions)]
        print(f"📊 After filtering: {len(df)} samples with valid emotions")
        
        print("\n🔄 Running predictions on entire dataset...")
        
        y_true = []
        y_pred = []
        confidences = []
        
        # Run predictions with progress bar
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Predicting"):
            text = row['cleaned_text']
            true_label = row['emotion_class']
            
            predicted, confidence = self.predict(text)
            
            y_true.append(true_label)
            y_pred.append(predicted)
            confidences.append(confidence)
        
        # Add predictions to dataframe
        df['predicted'] = y_pred
        df['confidence'] = confidences
        df['correct'] = df['predicted'] == df['emotion_class']
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        # Print results
        print("\n" + "="*60)
        print("MYANMAR EMOTION CLASSIFICATION MODEL - FULL CSV TEST")
        print("="*60)
        
        print(f"\n📊 DATASET INFO")
        print("-"*50)
        print(f"  Total samples: {len(df)}")
        print(f"  Emotions: {', '.join(self.emotions)}")
        
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
        
        print(f"\n📊 DETAILED BREAKDOWN BY EMOTION")
        print("-"*50)
        
        for emotion in self.emotions:
            emotion_data = df[df['emotion_class'] == emotion]
            if len(emotion_data) > 0:
                correct = emotion_data['correct'].sum()
                total = len(emotion_data)
                acc = correct / total
                avg_conf = emotion_data['confidence'].mean()
                print(f"  {emotion:10} : {int(correct)}/{total} = {acc:.1%}  (avg conf: {avg_conf:.1%})")
        
        # Show confusion matrix
        print(f"\n📊 CONFUSION MATRIX")
        print("-"*50)
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred, labels=self.emotions)
        
        # Print confusion matrix as table
        print(f"{'':12}", end="")
        for emotion in self.emotions:
            print(f"{emotion[:6]:<8}", end="")
        print()
        
        for i, emotion_true in enumerate(self.emotions):
            print(f"{emotion_true:<12}", end="")
            for j in range(len(self.emotions)):
                print(f"{cm[i][j]:<8}", end="")
            print()
        
        # Save results
        output_path = csv_path.replace('.csv', '_with_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"\n💾 Results saved to: {output_path}")
        
        # Show sample of misclassifications
        print(f"\n📝 SAMPLE MISCLASSIFICATIONS (First 10)")
        print("-"*50)
        misclassified = df[~df['correct']]
        if len(misclassified) > 0:
            for _, row in misclassified.head(10).iterrows():
                print(f"  Text: {row['cleaned_text'][:50]}...")
                print(f"    True: {row['emotion_class']} → Predicted: {row['predicted']} (conf: {row['confidence']:.1%})")
        else:
            print("  No misclassifications found!")
        
        print("\n" + "="*60)
        if accuracy >= 0.85:
            print("🏆 FINAL VERDICT: PRODUCTION READY")
        elif accuracy >= 0.75:
            print("✅ FINAL VERDICT: GOOD - READY FOR USE")
        elif accuracy >= 0.65:
            print("⚠️ FINAL VERDICT: ACCEPTABLE - CAN IMPROVE")
        else:
            print("❌ FINAL VERDICT: NEEDS MORE WORK")
        print("="*60)
        
        return accuracy, f1_macro, df
    
    def run_test(self):
        """Run test on the CSV file"""
        return self.test_on_csv("../../data/processed/dataset_processed.csv")

if __name__ == "__main__":
    tester = AccuracyTester()
    acc, f1, df = tester.run_test()