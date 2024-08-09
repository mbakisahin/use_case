from fastapi import HTTPException
from services.data_services.services import DataProcessingService

def process_data():
    """
    Process data using DataProcessingService.

    Returns:
        dict: Status of the data processing.

    Raises:
        HTTPException: If data processing fails.
    """
    try:
        data_processing_service = DataProcessingService()
        result = data_processing_service.process_data()
        return {"status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data processing failed: {str(e)}")
