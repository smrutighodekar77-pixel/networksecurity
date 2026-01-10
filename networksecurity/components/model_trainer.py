import os
import sys
import mlflow
import mlflow.sklearn
import dagshub

from urllib.parse import urlparse
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from networksecurity.entity.config_entity import ModelTrainerConfig

from networksecurity.utils.main_utils.utils import (
    save_object,
    load_object,
    load_numpy_array_data,
    evaluate_models,
)
from networksecurity.utils.ml_utils.metric.classification_metric import (
    get_classification_score,
)
from networksecurity.utils.ml_utils.model.estimator import NetworkModel


# DagsHub MLflow initialization (ONLY this)
dagshub.init(
    repo_owner="smrutighodekar77-pixel",
    repo_name="networksecurity",
    mlflow=True,
)


class ModelTrainer:
    def __init__(
        self,
        model_trainer_config: ModelTrainerConfig,
        data_transformation_artifact: DataTransformationArtifact,
    ):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def track_mlflow(self, model, train_metric, test_metric):
        mlflow.set_experiment("ModelTrainer-Experiment")

        with mlflow.start_run():

            mlflow.log_param("model_name", model.__class__.__name__)

            mlflow.log_metric("train_f1", train_metric.f1_score)
            mlflow.log_metric("train_precision", train_metric.precision_score)
            mlflow.log_metric("train_recall", train_metric.recall_score)

            mlflow.log_metric("test_f1", test_metric.f1_score)
            mlflow.log_metric("test_precision", test_metric.precision_score)
            mlflow.log_metric("test_recall", test_metric.recall_score)

            mlflow.sklearn.log_model(model, "model")

    def train_model(self, X_train, y_train, X_test, y_test):

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "AdaBoost": AdaBoostClassifier(),
        }

        params = {
            "Random Forest": {"n_estimators": [16, 32, 64, 128]},
            "Decision Tree": {"criterion": ["gini", "entropy"]},
            "Gradient Boosting": {
                "learning_rate": [0.1, 0.01],
                "n_estimators": [64, 128],
            },
            "Logistic Regression": {},
            "AdaBoost": {"n_estimators": [64, 128]},
        }

        model_report = evaluate_models(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            models=models,
            param=params,
        )

        best_model_score = max(model_report.values())
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        logging.info(f"Best model selected: {best_model_name}")

        best_model.fit(X_train, y_train)

        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)

        train_metric = get_classification_score(y_train, y_train_pred)
        test_metric = get_classification_score(y_test, y_test_pred)

        # MLflow tracking
        self.track_mlflow(best_model, train_metric, test_metric)

        # Load preprocessor
        preprocessor = load_object(
            self.data_transformation_artifact.transformed_object_file_path
        )

        # Create final network model
        network_model = NetworkModel(
            preprocessor=preprocessor,
            model=best_model,
        )

        # Model push (FINAL MODEL)
        os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
        save_object(
            self.model_trainer_config.trained_model_file_path,
            network_model,
        )

        os.makedirs("final_model", exist_ok=True)
        save_object("final_model/model.pkl", network_model)

        logging.info("Model pushed successfully")

        return ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=train_metric,
            test_metric_artifact=test_metric,
        )

    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            train_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_train_file_path
            )
            test_arr = load_numpy_array_data(
                self.data_transformation_artifact.transformed_test_file_path
            )

            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            return self.train_model(X_train, y_train, X_test, y_test)

        except Exception as e:
            raise NetworkSecurityException(e, sys)
