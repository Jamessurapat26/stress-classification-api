from fastapi import FastAPI
from endpoint.predictEndpoint import router as predict_router
from config.cors import add_cors_middleware

app = FastAPI()
add_cors_middleware(app)
app.include_router(predict_router, prefix="/predict", tags=["predict"])

@app.get("/")
def health_check():
    return {"status": "ok"}
