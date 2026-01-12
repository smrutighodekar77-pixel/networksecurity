import os
import sys
import io
import certifi
import pandas as pd
from dotenv import load_dotenv

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, JSONResponse

from networksecurity.utils.main_utils.utils import load_object
from networksecurity.utils.ml_utils.model.estimator import NetworkModel
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

import uvicorn

# Load environment variables
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")

# MongoDB (optional)
if mongo_db_url:
    import pymongo
    ca = certifi.where()
    client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
else:
    client = None
    logging.warning("MongoDB URL not found, skipping DB connection")

# FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

network_model = None

def load_model():
    global network_model
    try:
        preprocessor = load_object("final_model/preprocessor.pkl")
        model = load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor, model)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Model loading failed: {e}")
        network_model = None

load_model()

@app.get("/")
def index():
    return RedirectResponse(url="/docs")

@app.get("/train")
def train_route():
    try:
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()
        load_model()
        return {"status": "Training completed"}
    except Exception as e:
        raise NetworkSecurityException(e, sys)

@app.post("/predict")
async def predict_route(file: UploadFile = File(...)):
    try:
        if network_model is None:
            return JSONResponse(status_code=500, content={"error": "Model not loaded"})

        contents = await file.read()
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))

        predictions = network_model.predict(df)
        df["prediction"] = predictions

        os.makedirs("prediction_output", exist_ok=True)
        output_path = "prediction_output/output.csv"
        df.to_csv(output_path, index=False)

        return {
            "rows": len(df),
            "output_file": output_path,
            "predictions": predictions.tolist()
        }

    except Exception as e:
        logging.error(e)
        return JSONResponse(status_code=500, content={"error": str(e)})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)

# trigger pipeline
