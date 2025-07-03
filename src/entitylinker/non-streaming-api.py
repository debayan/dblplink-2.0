from flask import Flask, request, jsonify
from entity_linker import EntityLinker
import traceback

# Load configuration
config = {
    "elasticsearch": "http://localhost:9222",
    "sparql_endpoint": "http://localhost:8897/sparql"
}

# Initialize EntityLinker once
entity_linker = EntityLinker(config)

app = Flask(__name__)


@app.route("/link_entities", methods=["POST"])
def link_entities():
    data = request.get_json()
    text = data.get("question")

    if not text:
        return jsonify({"error": "Missing 'question' field in JSON body"}), 400

    try:
        spans = entity_linker.detect_spans_types(text)
        candidate_results = entity_linker.fetch_candidates(text, spans)

        entity_candidates = []
        for candidate in candidate_results:
            uris = [(item['_id'], item['_source']['label'], item['_source']['type']) for item in candidate]
            entity_candidates.append(uris)

        final_result = entity_linker.rerank_candidates(text, spans, entity_candidates)
        return jsonify(final_result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)

