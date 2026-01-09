import sys
from sklearn.metrics import f1_score, precision_score, recall_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact


def get_classification_score(y_true, y_pred) -> ClassificationMetricArtifact:
    try:
        f1 = f1_score(y_true, y_pred, average="weighted")
        precision = precision_score(y_true, y_pred, average="weighted")
        recall = recall_score(y_true, y_pred, average="weighted")

        return ClassificationMetricArtifact(
            f1_score=f1,
            precision_score=precision,
            recall_score=recall
        )

    except Exception as e:
        raise NetworkSecurityException(e, sys)
