import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# 1. Chargement des données
# ============================================================

DATA_PATH = Path("data/Student Placement.csv")

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("DIMENSIONS DU DATASET")
print("=" * 60)
print(f"Nombre d'observations : {df.shape[0]}")
print(f"Nombre de variables   : {df.shape[1]}")

# ============================================================
# 2. Aperçu des données
# ============================================================

print("\n" + "=" * 60)
print("APERÇU DES DONNÉES")
print("=" * 60)

print(df.head())

# ============================================================
# 3. Types des variables
# ============================================================

print("\n" + "=" * 60)
print("TYPES DES VARIABLES")
print("=" * 60)

print(df.dtypes)

# ============================================================
# 4. Valeurs manquantes
# ============================================================

print("\n" + "=" * 60)
print("VALEURS MANQUANTES")
print("=" * 60)

missing = df.isnull().sum()

print(missing)

# ============================================================
# 5. Doublons
# ============================================================

print("\n" + "=" * 60)
print("DOUBLONS")
print("=" * 60)

print(f"Nombre de doublons : {df.duplicated().sum()}")

# ============================================================
# 6. Statistiques descriptives
# ============================================================

print("\n" + "=" * 60)
print("STATISTIQUES DESCRIPTIVES")
print("=" * 60)

print(df.describe())

# ============================================================
# 7. Variables catégorielles
# ============================================================

categorical_columns = df.select_dtypes(include="object").columns

print("\n" + "=" * 60)
print("VARIABLES CATÉGORIELLES")
print("=" * 60)

for column in categorical_columns:
    print(f"\n--- {column} ---")
    print(df[column].value_counts())

# ============================================================
# 8. Distribution de la variable cible
# ============================================================

target = "Profile"

print("\n" + "=" * 60)
print("DISTRIBUTION DE LA VARIABLE CIBLE")
print("=" * 60)

print(df[target].value_counts())

# ============================================================
# 9. Histogrammes des variables quantitatives
# ============================================================

numeric_columns = df.select_dtypes(include="number").columns

df[numeric_columns].hist(
    figsize=(14, 10),
    bins=15
)

plt.suptitle(
    "Distribution des variables quantitatives",
    fontsize=16
)

plt.tight_layout()

plt.savefig(
    "distribution_variables.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ============================================================
# 10. Matrice de corrélation
# ============================================================

correlation = df[numeric_columns].corr()

print("\n" + "=" * 60)
print("MATRICE DE CORRÉLATION")
print("=" * 60)

print(correlation.round(2))

plt.figure(figsize=(12, 9))

plt.imshow(correlation, interpolation="nearest")
plt.colorbar()

plt.xticks(
    range(len(correlation.columns)),
    correlation.columns,
    rotation=90
)

plt.yticks(
    range(len(correlation.columns)),
    correlation.columns
)

plt.title("Matrice de corrélation")

plt.tight_layout()

plt.savefig(
    "matrice_correlation.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
