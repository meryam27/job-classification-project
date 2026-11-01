
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline

from pyspark.ml.feature import (
    Tokenizer,
    CountVectorizer,
    IDF,
    StringIndexer,
    IndexToString
)

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import concat_ws


# ============================================================
# 1. CREATION DE LA SESSION SPARK
# ============================================================

spark = SparkSession.builder \
    .appName("JobClassificationTraining") \
    .getOrCreate()


# ============================================================
# 2. CHARGEMENT DU DATASET
# ============================================================

df = spark.read.csv(
    "data/clean_offers.csv",
    header=True,
    inferSchema=True,
    multiLine=True,
    escape='"'
)

# Afficher le schéma pour vérifier les noms des colonnes
df.printSchema()


# ============================================================
# 3. PREPARATION DU TEXTE
# ============================================================

# On fusionne les trois colonnes textuelles.
# Chaque ligne représente une offre d'emploi.
# La colonne Category sera utilisée comme label,
# mais elle ne sera PAS incluse dans le texte.

df = df.withColumn(
    "text",
    concat_ws(" ", "Title", "Description", "Skills_Tags")
)


# ============================================================
# 4. DIVISION DU DATASET
# ============================================================

# 80% pour entraîner le modèle
# 20% pour tester ses performances

train_df, test_df = df.randomSplit(
    [0.8, 0.2],
    seed=42
)

print(f"Nombre total de lignes : {df.count()}")
print(f"Nombre de catégories : {df.select('Category').distinct().count()}")
print(f"Lignes d'entrainement : {train_df.count()}")
print(f"Lignes de test : {test_df.count()}")


# ============================================================
# 5. PIPELINE DE TRAITEMENT DU TEXTE
# ============================================================

# Cette partie s'occupe UNIQUEMENT du texte.
# Elle ne dépend pas de la colonne Category.

text_pipeline = Pipeline(stages=[

    # Etape 1 : Tokenization
    # Découper le texte en une liste de mots (tokens).

    Tokenizer(
        inputCol="text",
        outputCol="words"
    ),

    # Etape 2 : CountVectorizer
    # Construire un vocabulaire à partir des mots.
    # vocabSize = maximum 10 000 mots dans le vocabulaire.
    # minDF = garder les mots présents dans au moins 2 offres.

    CountVectorizer(
        inputCol="words",
        outputCol="rawFeatures",
        vocabSize=10000,
        minDF=2
    ),

    # Etape 3 : IDF
    # Calculer le poids de chaque mot.
    # Les mots fréquents dans toutes les offres
    # auront généralement un poids plus faible.

    IDF(
        inputCol="rawFeatures",
        outputCol="features"
    )

])


# ============================================================
# 6. APPRENTISSAGE DES TRANSFORMATIONS DU TEXTE
# ============================================================

# fit() apprend :
# - Le vocabulaire avec CountVectorizer.
# - Les poids IDF avec IDF.

# IMPORTANT :
# On utilise UNIQUEMENT train_df pour apprendre.
# On ne doit jamais apprendre le vocabulaire sur test_df.

text_model = text_pipeline.fit(train_df)


# ============================================================
# 7. TRANSFORMATION DES DONNEES
# ============================================================

# On applique les transformations apprises aux deux datasets.
# Le même vocabulaire et les mêmes poids IDF sont utilisés.

train_features = text_model.transform(train_df)

test_features = text_model.transform(test_df)


# ============================================================
# 8. ENCODAGE DES CATEGORIES
# ============================================================

# Logistic Regression ne travaille pas directement avec
# des catégories textuelles comme "Data Engineering".
# On doit donc les convertir en nombres.

label_indexer = StringIndexer(
    inputCol="Category",
    outputCol="label",
    handleInvalid="keep"
)

# Apprendre la correspondance catégorie -> nombre
# uniquement sur les données d'entrainement.

label_model = label_indexer.fit(train_df)


# Appliquer le même encodage aux deux datasets.

train_encoded = label_model.transform(train_features)

test_encoded = label_model.transform(test_features)


# ============================================================
# 9. DEFINITION DU CLASSIFICATEUR
# ============================================================

# Logistic Regression apprend la relation entre :
# - features : les vecteurs numériques du texte.
# - label : les catégories encodées en nombres.

lr = LogisticRegression(
    featuresCol="features",
    labelCol="label",
    maxIter=100,
    regParam=0.01
)


# ============================================================
# 10. ENTRAINEMENT DU CLASSIFICATEUR
# ============================================================

# Ici, on entraîne uniquement Logistic Regression.
# Le traitement du texte et l'encodage des catégories
# ont déjà été réalisés.

classifier_model = lr.fit(train_encoded)


# ============================================================
# 11. PREDICTION SUR LE DATASET DE TEST
# ============================================================

# Le classificateur prédit un label numérique
# pour chaque offre du dataset de test.

predictions = classifier_model.transform(test_encoded)


# ============================================================
# 12. EVALUATION DU MODELE
# ============================================================

evaluator = MulticlassClassificationEvaluator(
    labelCol="label",
    predictionCol="prediction"
)

# Accuracy : proportion des prédictions correctes.

accuracy = evaluator.setMetricName("accuracy").evaluate(predictions)

# F1-score pondéré : prend en compte la précision et le rappel
# de chaque catégorie, en tenant compte de sa fréquence.

f1 = evaluator.setMetricName("f1").evaluate(predictions)

print("\n========== RESULTATS DU MODELE ==========")

print(f"Accuracy : {accuracy:.4f}")

print(f"F1-score pondere : {f1:.4f}")


# ============================================================
# 13. AFFICHAGE DES PREDICTIONS
# ============================================================

# Pour comprendre les résultats, on affiche :
# - Category : vraie catégorie.
# - prediction : catégorie prédite sous forme numérique.
# - probability : probabilités attribuées aux classes.

predictions.select(
    "Category",
    "label",
    "prediction",
    "probability"
).show(truncate=False)


# ============================================================
# 14. CONVERSION DU LABEL PREDIT EN NOM DE CATEGORIE
# ============================================================

# Logistic Regression retourne un nombre.
# IndexToString permet de retrouver le nom de la catégorie.

converter = IndexToString(
    inputCol="prediction",
    outputCol="predictedCategory",
    labels=label_model.labels
)

predictions = converter.transform(predictions)

# Afficher la vraie catégorie et la catégorie prédite.

predictions.select(
    "Category",
    "predictedCategory",
    "prediction"
).show(truncate=False)


# ============================================================
# 15. SAUVEGARDE DES TROIS MODELES
# ============================================================

# Sauvegarder le modèle de traitement du texte.
# Il contient Tokenizer, CountVectorizerModel et IDFModel.

text_model.write().overwrite().save(
    "models/text_pipeline"
)

# Sauvegarder le classificateur entraîné.

classifier_model.write().overwrite().save(
    "models/classifier"
)

# Sauvegarder le modèle d'encodage des catégories.
# Il contient la correspondance entre les labels et les catégories.

label_model.write().overwrite().save(
    "models/label_indexer"
)

print("\nLes trois modeles ont ete sauvegardes avec succes.")


# ============================================================
# 16. FERMETURE DE LA SESSION SPARK
# ============================================================

spark.stop()
