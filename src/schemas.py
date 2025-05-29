from pydantic import BaseModel
from typing import List
from pydantic import Field
from typing import Literal
from pydantic import BaseModel, Field, model_validator
from typing import Literal
from src.feature_names import FEATURE_NAMES

class PredictRequest(BaseModel):
    features: dict[str, float]
    model_name: Literal["knn", "svm", "nb"] = Field(..., description="Model to use: 'knn', 'svm', or 'nb'")

    @model_validator(mode="after")
    def validate_features_keys(self) -> "PredictRequest":
        if self.features is None:
            raise ValueError("Missing 'features' dictionary.")

        feature_keys = set(self.features.keys())
        required_keys = set(FEATURE_NAMES)

        if feature_keys != required_keys:
            missing = required_keys - feature_keys
            extra = feature_keys - required_keys
            message_parts = []
            if missing:
                message_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                message_parts.append(f"Unexpected keys: {sorted(extra)}")
            raise ValueError("Invalid feature keys. " + "; ".join(message_parts))

        return self
    
class PredictResponse(BaseModel):
    predicted_class: str
    probabilities: dict

class PredictRequestAll(BaseModel):
    features: dict[str, float]

    @model_validator(mode="after")
    def validate_features_keys(self) -> "PredictRequest":
        if self.features is None:
            raise ValueError("Missing 'features' dictionary.")

        feature_keys = set(self.features.keys())
        required_keys = set(FEATURE_NAMES)

        if feature_keys != required_keys:
            missing = required_keys - feature_keys
            extra = feature_keys - required_keys
            message_parts = []
            if missing:
                message_parts.append(f"Missing keys: {sorted(missing)}")
            if extra:
                message_parts.append(f"Unexpected keys: {sorted(extra)}")
            raise ValueError("Invalid feature keys. " + "; ".join(message_parts))

        return self