# ============================================================
# EXPLORATION DES DONNEES
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def descriptive_statistics(df):
    """
    Retourne les statistiques descriptives.
    """
    return df.describe(include="all")


def missing_values(df):
    """
    Calcule le nombre de valeurs manquantes par variable.
    """
    return df.isna().sum().sort_values(ascending=False)


def correlation_matrix(df):
    """
    Calcule la matrice de corrélation
    pour les variables numériques.
    """
    numeric_df = df.select_dtypes(include="number")

    return numeric_df.corr()


def plot_correlation_matrix(df):
    """
    Affiche la matrice de corrélation.
    """

    correlation = correlation_matrix(df)

    plt.figure(figsize=(10, 7))

    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f"
    )

    plt.title("Matrice de corrélation")
    plt.tight_layout()

    plt.show()
