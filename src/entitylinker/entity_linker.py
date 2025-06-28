import sys,os,json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from elasticsearch import Elasticsearch
from candidate_reranker import CandidateReranker


class EntityLinker:    
    def __init__(self, config):
        self.config = config
        MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.es = Elasticsearch(config['elasticsearch'])
        self.candidate_reranker = CandidateReranker(self.model, self.tokenizer, config, self.device)

    def detect_spans_types(self, text):
        """
        Detects spans in the text and returns their types.
        This is a placeholder implementation.
        """
        messages = [
        {"role": "system", "content": "You are an information extraction assistant."},
        {"role": "user", "content": f"""Extract named entities from the following sentence and classify them into one of the following types: person, publication, venue.
         For example: "Which papers in ICLR 2023 were authored by Debayan Banerjee?" should return a person entity with type "person" and label "Debayan Banerjee" and venue entity with
         type "venue" and label "ICLR 2023".
        Return the result as a JSON array of objects with "type" and "label" fields.
        Sentence: "{text}"
        Entities:"""}
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        decoded_outputs = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output = decoded_outputs[0]
        json_match = re.search(r'\[\s*{.*?}\s*]', output, re.DOTALL)
        entities = []
        if json_match:
            json_str = json_match.group(0)
            try:
                entities = json.loads(json_str)
                print("Extracted entity list:")
                print(json.dumps(entities, indent=2))
            except json.JSONDecodeError as e:
                print("JSON decoding error:", e)
                print("Raw matched text:\n", json_str)
        else:
            print("No JSON array found in model output.")
        extracted_spans = entities
        # Placeholder for span detection logic
        return extracted_spans
    
    def fetch_candidates(self, text, spans):
        """
        Fetches candidate entities for a given span in the text.
        This is a placeholder implementation.
        """
        results = []
        for span in spans:
            entity_type = span['type']
            types = []
            if entity_type == "person":
                types = ["https://dblp.org/rdf/schema#Creator", "https://dblp.org/rdf/schema#Person"]
            elif entity_type == "publication":
                types = ["https://dblp.org/rdf/schema#Book", "https://dblp.org/rdf/schema#Article", "https://dblp.org/rdf/schema#Publication"]
            elif entity_type == "venue":
                types = ["https://dblp.org/rdf/schema#Conference", "https://dblp.org/rdf/schema#Incollection", "https://dblp.org/rdf/schema#Inproceedings", "https://dblp.org/rdf/schema#Journal", "https://dblp.org/rdf/schema#Series", "https://dblp.org/rdf/schema#Stream", "https://dblp.org/rdf/schema#Publication"]

            label = span['label']
            print(f"Fetching candidates for type: {entity_type}, label: {label}")    
            query = {
                "size": 10,
                "query": {
                    "bool": {
                        "must": [
                            {"terms": {"type": types}},   # exact match on type
                            {"match": {"label": label}}        # fuzzy/textual match on label
                        ]
                    }
                }
            }
            response = self.es.search(index='dblp', body=query)
            # Extract entity field from results
            results.append([
                hit
                for hit in response["hits"]["hits"]
            ])
        # Placeholder for candidate fetching logic
        return results
    
    def rerank_candidates(self, text, spans, entity_candidates):
        """
        Reranks the candidates based on some criteria.
        This is a placeholder implementation.
        """
        sorted_spans = self.candidate_reranker.rerank_candidates(text, spans, entity_candidates)
        return sorted_spans
    
if __name__ == "__main__":
    # Example usage
    config = {
        "elasticsearch": "http://localhost:9222",
        "sparql_endpoint": "http://localhost:7015"
    }
    
    entity_linker = EntityLinker(config)
    
    text = "which papers in sigir 2023 was authored by Chris Biemann?"
    print("Detecting spans and types in text:", text)
    spans = entity_linker.detect_spans_types(text)
    print("Detected Spans:", spans)
    print(" Fetching candidates for detected spans...")
    candidate_results = entity_linker.fetch_candidates(text, spans)
    entity_candidates = []
    for candidate in candidate_results:
        uris = []
        for item in candidate:
            uris.append((item['_id'], item['_source']['label']))
        entity_candidates.append(uris)
    print("Candidate Results:", entity_candidates)
    print("sorting candidates ...")
    sorted_spans = entity_linker.rerank_candidates(text, spans, entity_candidates)
    print("Final Reranked Entities:")
    for sorted_span in sorted_spans:
        print(f"Span: {sorted_span['span']}")
        print("Entities:")
        for entity_uri, score in sorted_span['entities']:
            print(f"  - {entity_uri}  (Score: {score})")