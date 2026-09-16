# ============================================================
# EVALUATION DES MODELES
# ============================================================

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test):

    predictions = model.predict(X_test)

    results = {

        "accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "precision":
            precision_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "recall":
            recall_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            ),

        "f1_score":
            f1_score(
                y_test,
                predictions,
                average="weighted",
                zero_division=0
            )
    }

    return results, predictions


def compare_models(results):

    comparison = pd.DataFrame(
        results
    ).T

    return comparison.sort_values(
        by="f1_score",
        ascending=False
    )
