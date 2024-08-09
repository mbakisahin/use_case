from fastapi import APIRouter
from services.data_services.app import process_data

router = APIRouter()

@router.post("/data")
def process_data_endpoint():
    """
    Trigger the data processing.

    Returns:
        dict: Confirmation message.
    """
    process_data()
    return {"key": "value"}

