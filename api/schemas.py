from pydantic import BaseModel
from typing import List

class EnergyInput(BaseModel):
    values: List[float]
