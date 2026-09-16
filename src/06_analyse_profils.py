import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Chargement
# ============================================================

df = pd.read_csv("data/Student Placement.csv")


# ============================================================
# Variables quantitatives
# ============================================================

numeric_columns = df.select_dtypes(
    include="number"
).columns


# ============================================================
# Moyennes par profil
# ============================================================

profile_means = df.groupby(
    "Profile"
)[numeric_columns].mean()


print("=" * 60)
print("MOYENNES DES VARIABLES PAR PROFIL")
print("=" * 60)

print(profile_means.round(2))


# ============================================================
# Export
# ============================================================

profile_means.to_csv(
    "results/moyennes_par_profil.csv"
)


# ============================================================
# Visualisation
# ============================================================

profile_means.T.plot(
    figsize=(14, 8),
    marker="o"
)

plt.title(
    "Caractéristiques moyennes selon le profil"
)

plt.xlabel("Variables")
plt.ylabel("Score moyen")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "results/moyennes_par_profil.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

print("\nAnalyse des profils terminée.")
