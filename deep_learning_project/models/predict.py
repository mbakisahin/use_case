from pydantic import BaseModel

class PredictRequest(BaseModel):
    """
    PredictRequest is a model that represents the input data required for predicting the amount of an item in a specific shop.

    Attributes:
    - shop_id (int): The unique identifier of the shop.
    - item_id (int): The unique identifier of the item.
    """
    shop_id: int
    item_id: int

