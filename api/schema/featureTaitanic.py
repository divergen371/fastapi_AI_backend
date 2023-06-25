from pydantic import BaseModel


class TitanicFeautureRequest(BaseModel):
    Sex: str
    Pclass: str
    Age: int
    Parch: int
    SibSp: int


class SurvivalProbabilityResopnse(BaseModel):
    survival_probability: float
