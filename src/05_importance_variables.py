import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# Chargement des données
# ============================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]


# ============================================================
# Modèle Random Forest
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)


# ============================================================
# Importance des variables
# ============================================================

importance = pd.DataFrame({
    "Variable": X_train.columns,
    "Importance": model.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

print("=" * 60)
print("IMPORTANCE DES VARIABLES")
print("=" * 60)

print(importance)


# ============================================================
# Graphique
# ============================================================

top_variables = importance.head(15)

plt.figure(figsize=(10, 7))

plt.barh(
    top_variables["Variable"][::-1],
    top_variables["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Variable")
plt.title("Importance des variables - Random Forest")

plt.tight_layout()

plt.savefig(
    "results/importance_variables.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Export
# ============================================================

importance.to_csv(
    "results/importance_variables.csv",
    index=False
)

print("\nAnalyse terminée.")
