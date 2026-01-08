import os
import sys
import pandas as pd
from scipy.stats import ks_2samp

from networksecurity.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact
)
from networksecurity.entity.config_entity import DataValidationConfig
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.constant.training_pipeline import SCHEMA_FILE_PATH
from networksecurity.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            logging.info("Initializing DataValidation")

            print("Schema path:", SCHEMA_FILE_PATH)
            print("Schema exists:", os.path.exists(SCHEMA_FILE_PATH))

            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            if self._schema_config is None:
                raise Exception("Schema YAML is empty or not loaded")

            logging.info("Schema loaded successfully")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that dataframe has all columns specified in the schema
        """
        try:
            # Your schema has columns as list of strings
            expected_columns = self._schema_config["columns"]

            logging.info(f"Expected columns count: {len(expected_columns)}")
            logging.info(f"Dataframe columns count: {len(dataframe.columns)}")

            missing_columns = set(expected_columns) - set(dataframe.columns)
            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}")
                return False

            return True

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold=0.05) -> bool:
        """
        Detect data drift using KS test and save drift report as YAML
        """
        try:
            status = True
            report = {}

            for column in base_df.columns:
                if base_df[column].dtype == "O":
                    continue

                ks_result = ks_2samp(base_df[column], current_df[column])

                # Convert to native Python types for YAML
                drift_found = bool(ks_result.pvalue < threshold)

                if drift_found:
                    status = False

                report[column] = {
                    "p_value": float(ks_result.pvalue),
                    "drift_status": drift_found
                }

            # Save drift report
            os.makedirs(os.path.dirname(self.data_validation_config.drift_report_file_path), exist_ok=True)
            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report
            )

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Execute full data validation workflow
        """
        try:
            logging.info("Starting data validation")

            train_df = self.read_data(self.data_ingestion_artifact.train_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            if not self.validate_number_of_columns(train_df):
                raise Exception("Train dataframe column validation failed")

            validation_status = self.detect_dataset_drift(base_df=train_df, current_df=test_df)

            # Save validated datasets
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False, header=True)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False, header=True)

            logging.info("Data validation completed successfully")

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
