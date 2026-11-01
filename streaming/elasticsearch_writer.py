
import os
from datetime import datetime, timezone

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

INDEX_NAME = "job_classifications"



client_options = {}
if "ElasticPassword2026!":
    client_options["basic_auth"] = (
        "elastic",
        "ElasticPassword2026!",
    )

es = Elasticsearch("http://localhost:9200", **client_options)


def check_elasticsearch_connection():
    if not es.ping():
        raise ConnectionError(
            f"Elasticsearch inaccessible ou authentification refusée à "
            f"{http://localhost:9200}. Vérifiez que le conteneur est démarré "
            "et définissez ELASTICSEARCH_USERNAME et ELASTICSEARCH_PASSWORD "
            "avec les identifiants configurés dans docker-compose.yml."
        )

    print("Connexion Elasticsearch réussie !")


def send_batch_to_elasticsearch(batch_df, batch_id):

    # Éviter de traiter un batch vide
    if batch_df.isEmpty():
        return

    actions = []

    # Récupération des lignes du micro-batch
    for row in batch_df.toLocalIterator():

        document = {
            "Title": row["Title"],
            "predicted_category": row["predicted_category"],
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "batch_id": batch_id
        }

        actions.append({
            "_index": INDEX_NAME,
            "_source": document
        })

    if actions:
        success, errors = bulk(
            es,
            actions,
            raise_on_error=False
        )

        print(
            f"Batch {batch_id} : "
            f"{success} documents envoyés à Elasticsearch"
        )

        if errors:
            print("Erreurs :", errors)
