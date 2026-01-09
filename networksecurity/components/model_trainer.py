import os
import sys
import mlflow
import mlflow.sklearn

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from networksecurity.entity.config_entity import ModelTrainerConfig
from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
)
from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.ensemble import RandomForestClassifier


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            # ✅ MLflow – correct & minimal
            mlflow.set_tracking_uri("file:./mlruns")
            mlflow.set_experiment("NetworkSecurity-Experiment")

        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info("Model training started")

            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = RandomForestClassifier(n_estimators=100, random_state=42)

            with mlflow.start_run():

                mlflow.log_param("model", "RandomForest")
                mlflow.log_param("n_estimators", 100)

                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)

                train_metric = get_classification_score(y_train, y_train_pred)
                test_metric = get_classification_score(y_test, y_test_pred)

                mlflow.log_metric("train_f1", train_metric.f1_score)
                mlflow.log_metric("test_f1", test_metric.f1_score)
                mlflow.log_metric("precision", test_metric.precision_score)
                mlflow.log_metric("recall", test_metric.recall_score)

                mlflow.sklearn.log_model(model, "model")

            preprocessor = load_object(
                self.data_transformation_artifact.transformed_object_file_path
            )

            network_model = NetworkModel(
                preprocessor=preprocessor,
                model=model
            )

            save_object(
                self.model_trainer_config.trained_model_file_path,
                network_model
            )

            logging.info("Model training completed successfully")

            return ModelTrainerArtifact(
                trained_model_file_path=self.model_trainer_config.trained_model_file_path,
                train_metric_artifact=train_metric,
                test_metric_artifact=test_metric,
            )

        except Exception as e:
            raise NetworkSecurityException(e, sys)
