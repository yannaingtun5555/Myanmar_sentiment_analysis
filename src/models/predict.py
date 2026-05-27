import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

class MyanmarEmotionClassifier:
    """Myanmar Sentiment Classification Model for 3 Classes"""
    
    def __init__(self, model_path: str = "./myanmar_sentiment_model_refined", 
                 confidence_threshold: float = 0.5):
        
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # Load metadata
        if os.path.exists(os.path.join(model_path, "metadata.json")):
            with open(os.path.join(model_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
                self.id2label = {int(k): v for k, v in metadata["id2label"].items()}
                self.label2id = metadata["label2id"]
        else:
            # Default for 3 classes
            self.id2label = {0: "positive", 1: "negative", 2: "neutral"}
            self.label2id = {"positive": 0, "negative": 1, "neutral": 2}
        
        self.confidence_threshold = confidence_threshold
        
        print(f"✅ Model loaded successfully")
        print(f"🏷️  Classes: {list(self.id2label.values())}")
        print(f"💻 Device: {self.device}")
    
    def predict(self, text: str) -> Tuple[str, float, Dict]:
        """Predict sentiment for single text"""
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=64
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            all_probs = probs[0].cpu().numpy()
            
            predicted_sentiment = self.id2label[predicted_class]
        
        # Apply threshold
        if confidence < self.confidence_threshold:
            return "neutral", confidence, {
                "positive": float(all_probs[0]),
                "negative": float(all_probs[1]),
                "neutral": float(all_probs[2])
            }
        
        return predicted_sentiment, confidence, {
            "positive": float(all_probs[0]),
            "negative": float(all_probs[1]),
            "neutral": float(all_probs[2])
        }
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict sentiment for multiple texts"""
        results = []
        for text in texts:
            sentiment, confidence, _ = self.predict(text)
            results.append((sentiment, confidence))
        return results
    
    def test_on_csv(self, csv_path: str, text_col: str = 'cleaned_text', 
                    label_col: str = 'emotion_class', save_results: bool = True):
        """Test model on CSV file and show metrics"""
        
        print(f"\n{'='*60}")
        print(f"TESTING ON CSV: {csv_path}")
        print(f"{'='*60}")
        
        # Load CSV
        df = pd.read_csv(csv_path)
        print(f"\nTotal samples: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        
        # Find text column
        if text_col not in df.columns:
            if 'text' in df.columns:
                text_col = 'text'
            elif 'cleaned_text' in df.columns:
                text_col = 'cleaned_text'
            else:
                text_col = df.columns[0]
        
        print(f"Using text column: '{text_col}'")
        
        # Clean data
        df = df.dropna(subset=[text_col])
        df[text_col] = df[text_col].astype(str).str.strip()
        df = df[df[text_col] != '']
        df = df[df[text_col] != 'nan']
        
        texts = df[text_col].tolist()
        
        # Check if labels exist
        has_labels = label_col in df.columns
        if has_labels:
            df[label_col] = df[label_col].astype(str).str.strip().str.lower()
            df = df[~df[label_col].isin(['nan', 'none', ''])]
            df = df[df[label_col].isin(self.label2id.keys())]
            df['true_label'] = df[label_col].map(self.label2id)
            true_labels = df['true_label'].tolist()
            texts = df[text_col].tolist()
            print(f"Valid samples with labels: {len(texts)}")
        
        # Predict
        print(f"\nPredicting {len(texts)} samples...")
        predictions = []
        confidences = []
        all_probs = []
        
        for text in texts:
            sentiment, confidence, probs = self.predict(text)
            predictions.append(sentiment)
            confidences.append(confidence)
            all_probs.append(probs)
        
        # Create results dataframe
        results_df = pd.DataFrame({
            'text': texts,
            'predicted': predictions,
            'confidence': confidences
        })
        
        if has_labels:
            results_df['true'] = df[label_col].values
            results_df['correct'] = results_df['predicted'] == results_df['true']
            
            # Calculate metrics
            pred_labels = [self.label2id[p] for p in predictions]
            accuracy = accuracy_score(true_labels, pred_labels)
            f1 = f1_score(true_labels, pred_labels, average='weighted')
            
            print(f"\n{'='*60}")
            print(f"📊 RESULTS")
            print(f"{'='*60}")
            print(f"Accuracy: {accuracy:.3f}")
            print(f"Weighted F1-Score: {f1:.3f}")
            
            print(f"\nClassification Report:")
            print(classification_report(true_labels, pred_labels, 
                                      target_names=list(self.label2id.keys())))
            
            # Confusion Matrix
            cm = confusion_matrix(true_labels, pred_labels)
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.label2id.keys(),
                       yticklabels=self.label2id.keys())
            plt.title(f'Confusion Matrix - {os.path.basename(csv_path)}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        
        # Show samples
        print(f"\n{'='*60}")
        print("SAMPLE PREDICTIONS (First 10)")
        print(f"{'='*60}")
        for i in range(min(10, len(results_df))):
            row = results_df.iloc[i]
            if has_labels:
                status = "✓" if row['correct'] else "✗"
                print(f"{status} Text: {row['text'][:50]}...")
                print(f"   Predicted: {row['predicted']} ({row['confidence']:.1%}) | True: {row['true']}")
            else:
                print(f"  Text: {row['text'][:50]}...")
                print(f"  Predicted: {row['predicted']} ({row['confidence']:.1%})")
            print()
        
        # Save results
        if save_results:
            output_path = csv_path.replace('.csv', '_predictions.csv')
            results_df.to_csv(output_path, index=False)
            print(f"✅ Results saved to: {output_path}")
        
        return results_df
    
    def test_single_texts(self):
        """Test on individual examples"""
        
        test_examples = [
            "ဒီနေ့ အရမ်းပျော်တယ်",
            "ငါ အရမ်းစိတ်ဆိုးတယ်",
            "ဒါကပုံမှန်ပါပဲ",
            "မင်းကိုချစ်တယ်",
            "ငါ့ဘဝအဆုံးသတ်ချင်တယ်",
            "ကျေးဇူးတင်ပါတယ်",
            "အဆင်ပြေပါတယ်",
            "မင်းကို မုန်းတယ်",
        ]
        
        print("\n" + "="*60)
        print("TESTING SINGLE TEXTS")
        print("="*60)
        
        for text in test_examples:
            sentiment, confidence, probs = self.predict(text)
            print(f"\nText: {text}")
            print(f"Predicted: {sentiment.upper()} ({confidence:.2%})")
            print(f"  Positive: {probs['positive']:.2%}")
            print(f"  Negative: {probs['negative']:.2%}")
            print(f"  Neutral: {probs['neutral']:.2%}")


# Run tests
if __name__ == "__main__":
    # Initialize classifier
    classifier = MyanmarEmotionClassifier(
        model_path="../../models/3_class/myanmar_sentiment_final",
        confidence_threshold=0.5
    )
    
    # Test on CSV file
    csv_file = "../../data/accuracy_test/test1_pro.csv"
    
    if csv_file and os.path.exists(csv_file):
        classifier.test_on_csv(csv_file, text_col='cleaned_text', label_col='emotion_class')
    else:
        print("\nNo CSV file provided, running example tests...")
        classifier.test_single_texts()
    
    # Interactive mode
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    print("Enter text to predict (type 'quit' to exit)")
    
    while True:
        text = input("\nEnter Burmese text: ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        if not text:
            continue
        
        sentiment, confidence, probs = classifier.predict(text)
        print(f"\nPrediction: {sentiment.upper()} (Confidence: {confidence:.2%})")
        print(f"  Positive: {probs['positive']:.2%}")
        print(f"  Negative: {probs['negative']:.2%}")
        print(f"  Neutral: {probs['neutral']:.2%}")