import argparse
from pathlib import Path

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import IndexToString, StringIndexerModel
from pyspark.sql import SparkSession
from pyspark.sql.functions import concat_ws


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Predire la categorie d'une offre d'emploi."
    )
    parser.add_argument("--title", required=True, help="Titre de l'offre.")
    parser.add_argument(
        "--description",
        default="",
        help="Description de l'offre.",
    )
    parser.add_argument(
        "--skills",
        default="",
        help="Competences et mots-cles de l'offre.",
    )
    args = parser.parse_args()

    if not any(
        value.strip()
        for value in (args.title, args.description, args.skills)
    ):
        parser.error("Fournissez au moins un texte dans l'offre.")

    project_dir = Path(__file__).resolve().parent
    model_paths = {
        "pipeline texte": project_dir / "models" / "text_pipeline",
        "classificateur": project_dir / "models" / "classifier",
        "indexeur de categories": project_dir / "models" / "label_indexer",
    }
    missing = [name for name, path in model_paths.items() if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            "Modèle(s) introuvable(s) : "
            + ", ".join(missing)
            + ". Exécutez d'abord training/train_model.py."
        )

    spark = (
        SparkSession.builder.appName("JobClassificationPrediction")
        .master("local[*]")
        .getOrCreate()
    )
    try:
        text_model = PipelineModel.load(str(model_paths["pipeline texte"]))
        classifier = LogisticRegressionModel.load(
            str(model_paths["classificateur"])
        )
        labels = StringIndexerModel.load(
            str(model_paths[            "indexeur de categories"])
        ).labels

        offer = spark.createDataFrame(
            [(args.title, args.description, args.skills)],
            ["Title", "Description", "Skills_Tags"],
        ).withColumn(
            "text",
            concat_ws(" ", "Title", "Description", "Skills_Tags"),
        )

        scored = classifier.transform(text_model.transform(offer))
        decoded = IndexToString(
            inputCol="prediction",
            outputCol="predictedCategory",
            labels=labels,
        ).transform(scored)
        result = decoded.select(
            "predictedCategory",
            "prediction",
            "probability",
        ).first()

        predicted_index = int(result["prediction"])
        confidence = float(result["probability"][predicted_index])
        print(f"Categorie predite : {result['predictedCategory']}")
        print(f"Confiance : {confidence:.2%}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
