# accuracy_report.py - Fixed version (no label_mapping.json required)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

class AccuracyTester:
    def __init__(self, model_path="../../models/trained/xlm-roberta-improved-91"):
        print("Loading model...")
        
        # Check if path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        
        # Get labels from model config (no label_mapping.json needed)
        self.id2label = self.model.config.id2label
        if self.id2label is None:
            # If no id2label in config, create default
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
    
    def run_test(self):
        test_cases = [
            # Joy
            ("ဒီ နေ့ အ ရမ်း ပျော် စရာ ကောင်း တယ်", "Joy"),
            ("ငါ့ မှာ အ ချိန် အ ရမ်း ကောင်း နေ တယ်", "Joy"),
            ("ဒီ ပွဲ က အ ရမ်း ကောင်း တယ်", "Joy"),
            
            # Love
            ("ငါ မင်း ကို ချစ် တယ်", "Love"),
            ("မင်း နဲ့ အ တူ ရှိ ရ တာ ပျော် တယ်", "Love"),
            ("ငါ့ ရဲ့ အ ချစ် ဆုံး", "Love"),
            
            # Anger
            ("ငါ အ ရမ်း ဒေါသ ထွက် တယ်", "Anger"),
            ("ဒါ က ငါ့ ကို စိတ် ဆိုး စေ တယ်", "Anger"),
            ("ငါ အ ရမ်း မကျေ နပ် ဘူး", "Anger"),
            
            # Fear
            ("ငါ အ ရမ်း ကြောက် နေ တယ်", "Fear"),
            ("ဒီ နေရာ က ငါ့ ကို ကြောက် စရာ ကောင်း တယ်", "Fear"),
            ("ငါ စိတ် ပူ နေ တယ်", "Fear"),
            
            # Sadness
            ("ငါ ဝမ်း နည်း နေ တယ်", "Sadness"),
            ("ဒီ နေ့ အ ရမ်း စိတ် ညစ် တယ်", "Sadness"),
            ("ငါ ငို နေ တယ်", "Sadness"),
            
            # Neutral
            ("ဒီ နေ့ ရာသီ ဥတု က သာ မာန် ပဲ", "Neutral"),
            ("အ လုပ် က ပုံ မ မှန် ပဲ", "Neutral"),
            ("ရုံး က အ စည်း အ ဝေး ပုံ မ မှန် ပါ ပဲ", "Neutral"),
            
            # Surprise
            ("အင်း ဒါ မျှော် လင့် ထား တာ မ ဟုတ် ဘူး", "Surprise"),
            ("ဒါ က ငါ့ ကို အံ့အား သင့် စေ တယ်", "Surprise"),
            ("အံ့သြ စရာ ပဲ", "Surprise"),
        ]
        
        y_true = []
        y_pred = []
        confidences = []
        
        print("Running predictions...\n")
        
        for text, expected in test_cases:
            predicted, confidence = self.predict(text)
            y_true.append(expected)
            y_pred.append(predicted)
            confidences.append(confidence)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print("="*50)
        print("MYANMAR EMOTION CLASSIFICATION MODEL")
        print("="*50)
        
        print(f"\nOVERALL PERFORMANCE")
        print("-"*50)
        print(f"  Accuracy:      {accuracy:.3f} ({accuracy:.1%})")
        print(f"  F1-Macro:      {f1_macro:.3f}")
        print(f"  F1-Weighted:   {f1_weighted:.3f}")
        print(f"  Avg Confidence: {np.mean(confidences):.3f} ({np.mean(confidences):.1%})")
        
        print(f"\nPER-CLASS PERFORMANCE")
        print("-"*50)
        report = classification_report(y_true, y_pred, target_names=self.emotions, digits=3)
        print(report)
        
        # Detailed breakdown
        print("\nDETAILED BREAKDOWN")
        print("-"*50)
        
        for emotion in self.emotions:
            indices = [i for i, e in enumerate(y_true) if e == emotion]
            if indices:
                correct = sum(1 for i in indices if y_pred[i] == emotion)
                total = len(indices)
                acc = correct / total if total > 0 else 0
                avg_conf = np.mean([confidences[i] for i in indices])
                print(f"  {emotion:10} : {correct}/{total} = {acc:.1%}  (avg conf: {avg_conf:.1%})")
        
        print("\n" + "="*50)
        print(f"FINAL VERDICT: {'PRODUCTION READY' if accuracy >= 0.85 else 'GOOD - CAN IMPROVE'}")
        print("="*50)
        
        return accuracy, f1_macro

if __name__ == "__main__":
    tester = AccuracyTester()
    acc, f1 = tester.run_test()