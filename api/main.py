from fastapi import FastAPI

from api.routers import feautureTitanic
app = FastAPI()

app.include_router(feautureTitanic.router)
