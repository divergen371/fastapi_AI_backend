from pydantic import BaseModel


class SchemaOfTitanicFeautureRequest(BaseModel):
    Sex: str
    Pclass: str
    Age: int
    Parch: int
    SibSp: int
