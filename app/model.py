import pickle
from app.config import get_settings
from fastapi import Depends
from tensorflow.keras import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model

class SpamClassifier:
    """
    Handles loading the spam detection model and tokenizer,
    preprocessing text input, and returning spam predictions.
    """

    def __init__(self):
        """
        Loads the trained model and tokenizer from disk using paths
        defined in environment variables.
        """
        settings = get_settings()
        self.model = load_model(settings.MODEL_PATH)

        with open(settings.TOKENIZER_PATH, "rb") as file:
            self.tokenizer = pickle.load(
                file
            )  # ← Fixed: changed from `pickle.loads` to `pickle.load`

    def preprocess(self, text: list[str]):
        """
        Tokenizes and pads input text for model consumption.

        Args:
            text (list[str]): A list of raw text strings.

        Returns:
            np.ndarray: Preprocessed and padded input ready for prediction.
        """
        sequences = self.tokenizer.texts_to_sequences(text)
        padded = pad_sequences(sequences, maxlen=100, padding="post")
        return padded

    def predict(self, text: str):
        """
        Predicts whether the given text is spam or ham.

        Args:
            text (str): The raw text message to classify.

        Returns:
            dict: A dictionary with `label` ("spam" or "ham") and `confidence` (float).
        """
        processed_input = self.preprocess([text])
        prediction = self.model.predict(processed_input)

        score = float(prediction[0][0])
        label = "spam" if score > 0.5 else "ham"
        confidence = score if label == "spam" else 1 - score

        return {"label": label, "confidence": round(confidence, 3)}

def get_model(settings = Depends(get_settings)):
    return SpamClassifier(settings.MODEL_PATH)