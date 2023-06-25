from fastapi import APIRouter
from api.model.featureTaitanic import SchemaOfTitanicFeautureRequest

router = APIRouter()


@router.get("/hello")
def get_hello():
    return {"message": "Hello World"}


@router.get("/api/{message}")
def get_any_message(message: str):
    return {"message": message}


@router.post("/api/tatanic")
def derive_score(request_body: SchemaOfTitanicFeautureRequest):
    return request_body
