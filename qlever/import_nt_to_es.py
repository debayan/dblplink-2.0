import re
import sys
from elasticsearch import Elasticsearch, helpers
from tqdm import tqdm

# --- Elasticsearch client ---
es = Elasticsearch("http://localhost:9222")

# Index name
INDEX_NAME = "dblp"

# --- Input file ---
nt_file = "dblp_labels_types.nt"

triple_pattern = re.compile(r'<(.+?)>\s+<(.+?)>\s+(?:<(.+?)>|"(.*?)")\s*\.')

# Temp store for entities
entity_data = {}

LABEL_PRED = "http://www.w3.org/2000/01/rdf-schema#label"
TYPE_PRED = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"

# Configurable batch size
BATCH_SIZE = 10000

with open(nt_file, "r", encoding="utf-8") as f:
    for line in tqdm(f, desc="Processing NT file"):
        match = triple_pattern.match(line)
        if not match:
            continue
        subj, pred, obj_uri, obj_literal = match.groups()
        if subj[0] == '_': #blanknode ignore
            continue

        if subj not in entity_data:
            entity_data[subj] = {"type": None, "label": None}

        if pred == LABEL_PRED:
            entity_data[subj]["label"] = obj_literal
        elif pred == TYPE_PRED:
            entity_data[subj]["type"] = obj_uri

count = 0
actions = []
for k,v in entity_data.items():
    count += 1
    actions.append({"_index": INDEX_NAME,"_id": k,"_source": v})
    if count%BATCH_SIZE == 0:
        helpers.bulk(es, actions)
        print(f"Indexed {len(actions)} documents.")
        actions = []
helpers.bulk(es, actions)
print(f"Indexed {len(actions)} documents.")

print("Done.")
