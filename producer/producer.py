import csv
import json
import time

from kafka import KafkaProducer

# 1. Connexion à Kafka
producer = KafkaProducer(
    bootstrap_servers="localhost:9092",

    value_serializer=lambda v: json.dumps(
        v
    ).encode("utf-8")
)

# 2. Lecture du dataset
with open(
    "data/job_offers.csv",
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    # 3. Envoyer chaque offre
    for row in reader:

        message = {
            "Title": row["Title"],
            "Description": row["Description"],
            "Skills_Tags": row["Skills_Tags"]
        }

        producer.send(
            "job_offers",
            value=message
        )

        print("Offre envoyée :", message["Title"])

        # Simuler une arrivée progressive
        time.sleep(2)

# 4. Attendre la confirmation des envois
producer.flush()

# 5. Fermer le Producer
producer.close()