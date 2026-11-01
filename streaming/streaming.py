import os
import sys
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType
)

from pyspark.sql.functions import (
    from_json,
    concat_ws
)

from pyspark.ml import PipelineModel
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.feature import StringIndexerModel

from pyspark.ml.feature import IndexToString
from elasticsearch_writer import (
    check_elasticsearch_connection,
    send_batch_to_elasticsearch,
)

sys.stdout.reconfigure(errors="backslashreplace")
python_executable = Path(sys.executable)
os.environ["PATH"] = (
    str(python_executable.parent) + os.pathsep + os.environ.get("PATH", "")
)
os.environ["PYSPARK_PYTHON"] = python_executable.name

project_dir = Path(__file__).resolve().parent.parent
kafka_connector = (
    "org.apache.spark:spark-sql-kafka-0-10_2.12:"
    f"{pyspark.__version__}"
)

check_elasticsearch_connection()

spark = SparkSession.builder \
    .appName("JobClassificationStreaming") \
    .master("local[*]") \
    .config("spark.jars.packages", kafka_connector) \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Connexion à Kafka
kafka_df = spark.readStream \
    .format("kafka") \
    .option(
        "kafka.bootstrap.servers",
        "localhost:9092"
    ) \
    .option(
        "subscribe",
        "job_offers"
    ) \
    .option(
        "startingOffsets",
        "latest"
    ) \
    .load()
    
# on s'interesse que le champ value de la réponse de kafka (JSON)
messages = kafka_df.select(
    col("value").cast("string").alias("json_data")
)

# transformer JSON en colonnes
schema = StructType([
    StructField("Title", StringType(), True),
    StructField("Description", StringType(), True),
    StructField("Skills_Tags", StringType(), True)
])
parsed_df = messages.select(
    from_json(
        col("json_data"),
        schema
    ).alias("data")
).select("data.*")

parsed_df = parsed_df.withColumn(
    "text",
    concat_ws(" ", "Title", "Description","Skills_Tags")
)

# Charger les transformations de texte
text_model = PipelineModel.load(
    str(project_dir / "models" / "text_pipeline")
)

# Charger le classificateur
classifier = LogisticRegressionModel.load(
    str(project_dir / "models" / "classifier")
)

# Charger le mapping des catégories
label_model = StringIndexerModel.load(
    str(project_dir / "models" / "label_indexer")
)

labels = label_model.labels

# transformation appliqués au text
features_df = text_model.transform(parsed_df)
#effectuer la prédiction
predictions = classifier.transform(features_df)

decoder = IndexToString(
    inputCol="prediction",
    outputCol="predicted_category",
    labels=labels
)

final_df = decoder.transform(predictions)


result_df = final_df.select(
    "Title",
    "predicted_category"
)


# structured streaming traite les données en micro-batch
def display_results(batch_df, batch_id):

    print(f"\n===== BATCH {batch_id} =====")

    batch_df.select(
        "Title",
        "predicted_category"
    ).show(truncate=80)
    
query = result_df.writeStream.foreachBatch(send_batch_to_elasticsearch).outputMode("append").option(
        "checkpointLocation",
        str(project_dir / "checkpoints" / "job_classification")
    ).start()

query.awaitTermination()