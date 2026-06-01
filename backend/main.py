from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

try:
    from backend.model import predict_image
except ImportError:
    from model import predict_image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Read image bytes
    img_bytes = await file.read()

    # Get prediction
    result = predict_image(img_bytes)

    return {
        "filename"  : file.filename,
        "prediction": result["class"],
        "confidence": result["confidence"]
    }

@app.get("/")
def root():
    return {"status": "Dog Cat API running ✅"}