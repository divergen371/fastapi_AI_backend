from fastapi import APIRouter
from ..machine_leaning.titanic import PredictOnAPI
from api.schema.featureTaitanic import (
    SurvivalProbabilityResopnse,
    TitanicFeautureRequest,
)

router = APIRouter()


@router.get("/hello")
def get_hello():
    return {"message": "Hello World"}


@router.get("/api/{message}")
def get_any_message(message: str):
    return {"message": message}


@router.post("/api/titanic", response_model=SurvivalProbabilityResopnse)
def derive_score(request_body: TitanicFeautureRequest):
    features_dict = request_body.__dict__
    survival_probability = PredictOnAPI.derive_survival_probablity(
        **features_dict
    )
    return {"survival_probability": survival_probability}
