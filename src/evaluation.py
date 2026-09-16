import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ============================================================
# Chargement
# ============================================================

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")

y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

# ============================================================
# Modèle
# ============================================================

model = RandomForestClassifier(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("=" * 60)
print("ÉVALUATION DU RANDOM FOREST")
print("=" * 60)

print(f"\nAccuracy : {accuracy:.4f}")

# ============================================================
# Classification report
# ============================================================

print("\n--- Classification report ---")

print(
    classification_report(
        y_test,
        predictions
    )
)

# ============================================================
# Matrice de confusion
# ============================================================

cm = confusion_matrix(
    y_test,
    predictions
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm
)

disp.plot()

plt.title("Matrice de confusion - Random Forest")

plt.tight_layout()

plt.savefig(
    "results/matrice_confusion.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nMatrice de confusion enregistrée.")
