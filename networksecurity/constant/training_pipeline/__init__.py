import os
import numpy as np

TARGET_COLUMN = "Result"

PIPELINE_NAME = "NetworkSecurity"
ARTIFACT_DIR = "Artifacts"

FILE_NAME = "phisingData.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"

ROOT_DIR = os.getcwd()

SCHEMA_FILE_PATH = os.path.join(
    ROOT_DIR,
    "data_schema",
    "schema.yaml"
)

SAVED_MODEL_DIR = "saved_models"
MODEL_FILE_NAME = "model.pkl"


DATA_INGESTION_COLLECTION_NAME = "NetworkData"
DATA_INGESTION_DATABASE_NAME = "SMRUTI"

DATA_INGESTION_DIR_NAME = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR = "feature_store"
DATA_INGESTION_INGESTED_DIR = "ingested"

DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO = 0.2


DATA_VALIDATION_DIR_NAME = "data_validation"
DATA_VALIDATION_VALID_DIR = "valid"
DATA_VALIDATION_INVALID_DIR = "invalid"

DATA_VALIDATION_DRIFT_REPORT_DIR = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME = "report.yaml"

PREPROCESSING_OBJECT_FILE_NAME = "preprocessing.pkl"


"""
Data Transformation related constant start with DATA_TRANSFORMATION VAR NAME
"""
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"

## kkn imputer to replace nan values
DATA_TRANSFORMATION_IMPUTER_PARAMS: dict = {
    "missing_values": np.nan,
    "n_neighbors": 3,
    "weights": "uniform",
}