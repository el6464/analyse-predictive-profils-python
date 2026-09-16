import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

# ============================================================
# Chargement des données préparées
# ============================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

# ============================================================
# Modèles
# ============================================================

models = {
    "Régression logistique": LogisticRegression(
        max_iter=2000
    ),

    "Arbre de décision": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(
        n_neighbors=5
    )
}

# ============================================================
# Entraînement et comparaison
# ============================================================

results = []

for name, model in models.items():

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Modèle": name,
        "Accuracy": accuracy
    })

    print("=" * 50)
    print(name)
    print(f"Accuracy : {accuracy:.4f}")

# ============================================================
# Résultats
# ============================================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)

print("\n" + "=" * 50)
print("COMPARAISON DES MODÈLES")
print("=" * 50)

print(results_df)

results_df.to_csv(
    "results/comparaison_modeles.csv",
    index=False
)
