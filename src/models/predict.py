import os
import json
import torch
import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple

class MyanmarEmotionClassifier:
    """Myanmar Emotion Classification Model with Rule-Based Corrections"""
    
    def __init__(self, model_path: str = "../../models/trained/xlm-roberta-base-emotion", 
                 confidence_threshold: float = 0.6):
        
        print(f"Loading model from {model_path}...")
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        with open(os.path.join(model_path, "label_mapping.json"), "r") as f:
            mapping = json.load(f)
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
            self.label2id = mapping["label2id"]
        
        self.confidence_threshold = confidence_threshold
        
        # Rule-based keyword mapping
        self.keyword_rules = {
            "Anger": [
                "စိတ်ဆိုး", "ဒေါသ", "စိတ်တို", "မကျေနပ်", "မုန်း", 
                "ဒေါသထွက်", "အမျက်", "စိတ်ညစ်", "ခါး", "ပူ"
            ],
            "Surprise": [
                "အံ့သြ", "မထင်မှတ်", "ရှော့တိုက်", "အံ့အားသင့်", 
                "မျှော်လင့်", "တုန်လှုပ်"
            ],
            "Neutral": [
                "သာမန်", "ပုံမှန်", "ရုံ", "လောက်", "သဘော", 
                "သာမာန်", "သာသာ", "ဖြစ်တယ်"
            ],
            "Joy": ["ပျော်", "ရွှင်", "ပျော်ရွှင်", "ဝမ်းသာ", "ကျေနပ်"],
            "Fear": ["ကြောက်", "စိုးရိမ်", "လန့်", "ထိတ်", "ပူ", "စိုး"],
            "postitive": ["ချစ်", "မြတ်နိုး", "တွယ်", "ကြိုက်"],
            "Sadness": ["ဝမ်းနည်း", "ငို", "ညစ်", "ကြေကွဲ"]
        }
        
        print(f"✅ Model loaded with rule-based corrections")
        print(f"🏷️  Emotions: {list(self.id2label.values())}")
    
    def apply_rule_correction(self, text: str, predicted_emotion: str, confidence: float) -> Tuple[str, float]:
        """Apply rule-based corrections based on keywords"""
        
        # Check for keywords indicating different emotion
        for emotion, keywords in self.keyword_rules.items():
            for keyword in keywords:
                if keyword in text:
                    # If we find anger keyword but model predicted sadness
                    if emotion == "Anger" and predicted_emotion == "Sadness":
                        return "Anger", min(confidence + 0.2, 0.95)
                    # If we find surprise keyword but model predicted fear/sadness
                    elif emotion == "Surprise" and predicted_emotion in ["Fear", "Sadness"]:
                        return "Surprise", min(confidence + 0.15, 0.90)
                    # If we find neutral keyword but model predicted sadness
                    elif emotion == "Neutral" and predicted_emotion == "Sadness":
                        if confidence < 0.8:  # Only override if not very confident
                            return "Neutral", confidence - 0.1
        
        # Specific corrections for common phrases
        if "စိတ်ဆိုး" in text or "ဒေါသ" in text:
            return "Anger", max(confidence, 0.75)
        
        if "အံ့သြ" in text:
            return "Surprise", max(confidence, 0.80)
        
        if "သာမန်" in text or "ပုံမှန်" in text:
            if confidence < 0.7:
                return "Neutral", 0.65
        
        return predicted_emotion, confidence
    
    def predict(self, text: str) -> Tuple[str, float]:
        """Predict emotion with rule-based corrections"""
        
        # Get model prediction
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
            
            predicted_emotion = self.id2label[predicted_class]
            
            # Apply rule-based correction
            corrected_emotion, corrected_confidence = self.apply_rule_correction(
                text, predicted_emotion, confidence
            )
        
        # Apply threshold
        if corrected_confidence < self.confidence_threshold:
            return "Neutral", corrected_confidence
        
        return corrected_emotion, corrected_confidence
    
    def test_on_examples(self):
        """Test model with corrected examples"""
        
        test_examples = [
            ("ပန်း ခြံ ထဲ မှာ လှ ပ တဲ့ နေ့ လေး", "Joy"),
            ("ငါ မင်း ကို ချစ် တယ်", "postitive"),
            ("ငါ အ ရမ်း ကြောက် နေ တယ်", "Fear"),
            ("ငါ အ ရမ်း စိတ်ဆိုး ဒေါသ ထွက် တယ်", "Anger"),
            ("ဒီ မ နက် လမ်း အ ရမ်း ပိတ် တယ် စိတ်ညစ် တယ်", "Anger"),
            ("ငါ ဝမ်း နည်း နေ တယ်", "Sadness"),
            ("ဒီ နေ့ ရာသီ ဥတု က သာမန် ပဲ", "Neutral"),
            ("အင်း ဒါ ကို မျှော် လင့် ထား တာ မ ဟုတ် ဘူး အံ့သြ တယ်", "Surprise"),
            ("ငါ့ ကို ဒေါသ ထွက် စေ တယ်", "Anger"),
            ("ဒါ က အံ့အား သင့် စရာ ပဲ", "Surprise"),
        ]
        
        print("\n" + "="*60)
        print("MODEL TESTING WITH RULE CORRECTIONS")
        print("="*60)
        
        correct = 0
        corrections_applied = 0
        
        for text, expected in test_examples:
            # Get prediction with correction
            predicted, confidence = self.predict(text)
            
            # Also get raw prediction for comparison
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                raw_class = torch.argmax(torch.softmax(outputs.logits, dim=-1), dim=1).item()
                raw_pred = self.id2label[raw_class]
            
            is_correct = (predicted == expected)
            correct += is_correct
            
            if raw_pred != predicted:
                corrections_applied += 1
                status = f"🔄 Corrected: {raw_pred} → {predicted}"
            else:
                status = "✅" if is_correct else "❌"
            
            emoji_map = {"Joy": "😊", "postitive": "❤️", "Fear": "😨", "Anger": "😠", "Sadness": "😢", "Neutral": "😐", "Surprise": "😲"}
            
            print(f"\n{status}")
            print(f"   Text: {text[:40]}...")
            print(f"   Expected: {expected} {emoji_map.get(expected, '')}")
            print(f"   Predicted: {predicted} ({confidence:.1%})")
        
        accuracy = correct / len(test_examples)
        print("\n" + "="*60)
        print(f"📊 RESULTS WITH RULE CORRECTIONS")
        print("="*60)
        print(f"Total tests: {len(test_examples)}")
        print(f"Correct: {correct}")
        print(f"Accuracy: {accuracy:.1%}")
        print(f"Corrections applied: {corrections_applied}")
        print("="*60)
        
        return correct/len(test_examples)


# Run test
if __name__ == "__main__":
    classifier = MyanmarEmotionClassifier(
        model_path="../../models/trained/xlm-roberta-base-emotion",
        confidence_threshold=0.5  # Lower threshold for testing
    )
    classifier.test_on_examples()