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
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        try:
            required_columns = self._schema_config["columns"]
            return len(dataframe.columns) == len(required_columns)
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def detect_dataset_drift(self, base_df, current_df, threshold=0.05) -> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                if base_df[column].dtype == "O":
                    continue

                ks_result = ks_2samp(
                    base_df[column],
                    current_df[column]
                )

                drift_found = ks_result.pvalue < threshold
                if drift_found:
                    status = False

                report[column] = {
                    "p_value": float(ks_result.pvalue),
                    "drift_status": drift_found
                }

            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.drift_report_file_path
                ),
                exist_ok=True
            )

            write_yaml_file(
                file_path=self.data_validation_config.drift_report_file_path,
                content=report
            )

            return status

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(
                self.data_ingestion_artifact.train_file_path
            )
            test_df = self.read_data(
                self.data_ingestion_artifact.test_file_path
            )

            if not self.validate_number_of_columns(train_df):
                raise Exception("Invalid column count")

            validation_status = self.detect_dataset_drift(
                base_df=train_df,
                current_df=test_df
            )

            os.makedirs(
                os.path.dirname(
                    self.data_validation_config.valid_train_file_path
                ),
                exist_ok=True
            )

            train_df.to_csv(
                self.data_validation_config.valid_train_file_path,
                index=False,
                header=True
            )

            test_df.to_csv(
                self.data_validation_config.valid_test_file_path,
                index=False,
                header=True
            )

            return DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
