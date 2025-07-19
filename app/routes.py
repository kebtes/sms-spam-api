from fastapi import APIRouter, Depends
from pydantic import BaseModel
from .model import SpamClassifier, get_model

router = APIRouter()


class SMSRequest(BaseModel):
    """
    Schema for the incoming SMS classification request.

    Attributes:
        message (str): The SMS text to be classified.
    """

    message: str


@router.post("/predict")
def predict(sms: SMSRequest, model: SpamClassifier = Depends(get_model)):
    """
    Predict whether an SMS message is spam or not.

    Args:
        sms (SMSRequest): The request body containing the SMS message.
        model (SpamClassifier): The spam classifier instance injected via FastAPI's dependency system.

    Returns:
        dict: A dictionary with the predicted label ('spam' or 'ham') and a confidence score.
    """
    return model.predict(sms.message)
