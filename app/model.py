import pickle
from app.config import get_settings
from fastapi import Depends
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

class SpamClassifier:
    def __init__(self):
        self.model = load_model(get_settings().MODEL_PATH)

        with open(get_settings().TOKENIZER_PATH, "rb") as file:
            self.tokenizer = pickle.loads(file)

    def preprocess(self, text: list[str]):
        sequences = self.tokenizer.texts_to_sequences(text)
        padded = pad_sequences(
            sequences,
            maxlen=100,
            padding="post"
        )

        return padded

    def predict(self, text: str):
        processed_input = self.preprocess([text])
        prediction = self.model.predict(processed_input)
        
        label = "spam" if prediction[0][0] > 0.5 else "ham"
        confidence = float(prediction[0][0]) if label == "spam" else 1 - float(prediction[0][0])
        
        return {
            "label": label,
            "confidence": round(confidence, 3)
        }
        
def get_model(settings = Depends(get_settings)):
    return SpamClassifier(settings.MODEL_PATH)