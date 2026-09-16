import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ============================================================
# Configuration
# ============================================================

DATA_PATH = Path("data/Student Placement.csv")

# ============================================================
# Chargement
# ============================================================

df = pd.read_csv(DATA_PATH)

# ============================================================
# Suppression des doublons
# ============================================================

df = df.drop_duplicates()

# ============================================================
# Variables
# ============================================================

target = "Profile"

X = df.drop(columns=[target])
y = df[target]

# ============================================================
# Encodage des variables catégorielles
# ============================================================

categorical_columns = X.select_dtypes(include="object").columns

X = pd.get_dummies(
    X,
    columns=categorical_columns,
    drop_first=False
)

# ============================================================
# Encodage de la variable cible
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

# ============================================================
# Séparation train / test
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

# ============================================================
# Sauvegarde
# ============================================================

X_train.to_csv(
    "X_train.csv",
    index=False
)

X_test.to_csv(
    "X_test.csv",
    index=False
)

pd.DataFrame({"Profile": y_train}).to_csv(
    "y_train.csv",
    index=False
)

pd.DataFrame({"Profile": y_test}).to_csv(
    "y_test.csv",
    index=False
)

print("Préparation terminée.")
print(f"Train : {X_train.shape}")
print(f"Test  : {X_test.shape}")

print("\nProfils :")
for i, label in enumerate(label_encoder.classes_):
    print(i, "=", label)
