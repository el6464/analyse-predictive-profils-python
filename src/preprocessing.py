# ============================================================
# PREPROCESSING
# Analyse prédictive des profils et compétences
# ============================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(path):
    """
    Charge le jeu de données.
    """
    return pd.read_csv(path)


def basic_cleaning(df):
    """
    Nettoyage de base :
    - suppression des doublons
    - suppression des lignes entièrement vides
    """
    df = df.copy()

    df = df.drop_duplicates()
    df = df.dropna(how="all")

    return df


def encode_categorical_variables(df):
    """
    Encode les variables catégorielles.
    """
    df = df.copy()

    encoders = {}

    categorical_columns = df.select_dtypes(
        include=["object", "category"]
    ).columns

    for column in categorical_columns:

        encoder = LabelEncoder()

        df[column] = encoder.fit_transform(
            df[column].astype(str)
        )

        encoders[column] = encoder

    return df, encoders


def split_data(df, target_column, test_size=0.2):
    """
    Sépare les variables explicatives et la variable cible,
    puis crée les ensembles d'entraînement et de test.
    """

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test
