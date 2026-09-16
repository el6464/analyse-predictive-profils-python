# Analyse prédictive des profils étudiants avec Python

## Présentation

Ce projet porte sur l'analyse et la classification de profils d'étudiants à partir de leurs compétences techniques, aptitudes et compétences déclarées.

L'objectif est d'explorer les données, d'identifier les principales relations entre les variables et de comparer plusieurs modèles de classification afin de prédire le profil d'un étudiant.

## Données

Le jeu de données contient 707 observations et 13 variables.

Les variables comprennent notamment :

- DSA
- DBMS
- OS
- CN
- Mathmetics
- Aptitute
- Comm
- Problem Solving
- Creative
- Hackathons
- Skill 1
- Skill 2

La variable cible est `Profile`.

Elle correspond à différents profils professionnels dans le domaine technologique.

## Méthodologie

### 1. Exploration

- Analyse de la structure des données
- Vérification des valeurs manquantes
- Détection des doublons
- Statistiques descriptives
- Analyse des variables catégorielles
- Analyse des corrélations
- Visualisation des distributions

### 2. Préparation

- Nettoyage des données
- Encodage des variables catégorielles
- Encodage de la variable cible
- Séparation des données en échantillons d'entraînement et de test

### 3. Modélisation

Plusieurs algorithmes de classification sont comparés :

- Régression logistique
- Arbre de décision
- Random Forest
- K-Nearest Neighbors

### 4. Évaluation

Les modèles sont évalués à l'aide de :

- Accuracy
- Precision
- Recall
- F1-score
- Matrice de confusion

## Résultats

Les performances des différents modèles sont comparées afin d'identifier leurs différences et leurs limites sur cette problématique de classification multi-classe.

Les résultats détaillés sont disponibles dans le dossier `results/`.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Structure du projet

```text
analyse-predictive-profils-python/
│
├── data/
│   └── Student Placement.csv
│
├── src/
│   ├── 01_exploration.py
│   ├── 02_preprocessing.py
│   ├── 03_modelisation.py
│   └── 04_evaluation.py
│
├── results/
│   ├── distribution_variables.png
│   ├── matrice_correlation.png
│   ├── comparaison_modeles.csv
│   └── matrice_confusion.png
│
├── README.md
├── requirements.txt
└── .gitignore
