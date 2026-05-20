import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MyanmarEmotionClassifier:
    """Wrapper for Myanmar emotion classification model"""
    
    def __init__(self, model_path: str = "models/trained/xlm-roberta-base-emotion"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.id2label = None
        self.label2id = None
        self.load_model()
    
    def load_model(self):
        """Load model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mapping
            with open(os.path.join(self.model_path, "label_mapping.json"), "r") as f:
                mapping = json.load(f)
                self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
                self.label2id = mapping["label2id"]
            
            logger.info(f"Model loaded successfully with {len(self.id2label)} emotions")
            logger.info(f"Emotions: {list(self.id2label.values())}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, text: str) -> Tuple[str, float, Dict[str, float]]:
        """Predict emotion for single text"""
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
            
            # Get all probabilities
            all_probs = {self.id2label[i]: probs[0][i].item() for i in range(len(self.id2label))}
        
        return self.id2label[predicted_class], confidence, all_probs
    
    def predict_batch(self, texts: List[str]) -> List[Tuple[str, float]]:
        """Predict emotions for multiple texts"""
        results = []
        for text in texts:
            emotion, confidence, _ = self.predict(text)
            results.append((emotion, confidence))
        return results
    
    def get_model_info(self) -> Dict:
        """Return model information"""
        return {
            "model_path": self.model_path,
            "device": str(self.device),
            "num_emotions": len(self.id2label),
            "emotions": list(self.id2label.values()),
            "model_type": "xlm-roberta-base",
            "accuracy": 0.847,  # From training
            "f1_score": 0.847
        }

# Singleton instance for fast loading
_model_instance = None

def get_model():
    """Get or create model singleton"""
    global _model_instance
    if _model_instance is None:
        _model_instance = MyanmarEmotionClassifier()
    return _model_instance