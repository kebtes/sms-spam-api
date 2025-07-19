from fastapi import APIRouter, Depends
from pydantic import BaseModel
from .model import SpamClassifier, get_model

router = APIRouter()

class SMSRequest(BaseModel):
    message: str

@router.post("/predict")
def predict(
    sms: SMSRequest,
    model: SpamClassifier = Depends(get_model)
):
    return model.predict(sms.message)