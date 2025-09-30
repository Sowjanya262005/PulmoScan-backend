from pydantic import BaseModel
from typing import List, Optional

class Prediction(BaseModel):
    disease: str
    label: str
    score: float
    topk_labels: List[str]
    topk_scores: List[float]
    heatmap_b64: Optional[str] = None
