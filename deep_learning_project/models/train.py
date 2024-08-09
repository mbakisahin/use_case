from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class LayerConfig(BaseModel):
    output_size: int
    activation: Optional[str]

    @field_validator('output_size')
    def check_output_size(cls, v):
        if v <= 0:
            raise ValueError('output_size must be greater than 0')
        return v

    @field_validator('activation')
    def check_activation(cls, v):
        if v is None or v.strip() == "":
            raise ValueError('activation must be a valid string')
        return v

class TrainRequest(BaseModel):
    batch_size: int = Field(default=8196, ge=1, le=9999, description="Batch size for training.")
    time_step: int = Field(default=10, ge=1, le=99, description="Time step for the model.")
    learning_rate: float = Field(default=0.01, gt=0, lt=1.0, description="Learning rate for the model.")
    epochs: int = Field(default=10, ge=1, le=999, description="Number of epochs for training.")
    n_components: int = Field(default=10, ge=1, le=49, description="Number of components.")
    train_ratio: float = Field(default=0.8, gt=0, lt=1.0, description="Ratio of the data used for training.")
    layer_architecture: List[LayerConfig]

    @field_validator('layer_architecture', mode='before')
    def validate_layer_architecture(cls, v):
        for layer in v:
            if layer['output_size'] <= 0:
                raise ValueError('output_size must be greater than 0')
            if layer['activation'] is None or layer['activation'].strip() == "":
                raise ValueError('activation must be a valid string')
        return v
