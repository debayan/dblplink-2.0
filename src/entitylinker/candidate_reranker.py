import json
import requests
from typing import List, Tuple
from urllib.parse import urlencode
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessorList
import torch.nn.functional as F
import numpy as np

class CandidateReranker:
    def __init__(self, model, tokenizer, config, device="cuda"):
        self.config = config
        self.endpoint = config["sparql_endpoint"]
        self.headers = {
            "Accept": "application/sparql-results+json"
        }
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def format_input(self, mention, context, entity_name, entity_info_line):
        entity_info_text = entity_info_line
        #print(entity_info_text)
        return f"""You are an assistant linking mentions to entities.
                   Document: {context}
                   Mention: {mention}
                   Candidate Entity: {entity_name}
                   Entity Info: {entity_info_text}
                   Question: Does the mention belong to this entity? Answer-Yes/No.
                   Answer:"""

    
    def compute_yes_score(self, mention, context, entity_name, entity_info_lines):
        # Add " Yes" to each input
        full_inputs = [self.format_input(mention, context, entity_name, entity_info_line) + " Yes" for entity_info_line in entity_info_lines]
        #print("Full Inputs for scoring:", full_inputs)
        inputs = self.tokenizer(full_inputs, return_tensors='pt', padding=True, truncation=True, max_length=128).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # (batch_size, seq_len, vocab_size)

        log_probs = F.log_softmax(logits, dim=-1)
        yes_token_id = self.tokenizer(" Yes", add_special_tokens=False)["input_ids"][0]

        # Find the log prob of the token " Yes" in each sequence
        scores = []
        for i in range(len(full_inputs)):
            # Find the position of the " Yes" token in input_ids
            yes_pos = (inputs.input_ids[i] == yes_token_id).nonzero(as_tuple=True)[0]
            if len(yes_pos) == 0:
                # fallback to second last token if " Yes" is not found
                yes_logprob = log_probs[i, -2, yes_token_id].item()
            else:
                yes_logprob = log_probs[i, yes_pos[0], yes_token_id].item()
            scores.append(yes_logprob)
        max_score = max(scores)
        best_index = scores.index(max_score)
        best_sentence = entity_info_lines[best_index]
        # print("entity", entity_name)
        # print("entity info line", entity_info_lines)
        # print("best sentence", best_sentence)
        return max_score, best_sentence

    def fetch_one_hop(self, entity_uri):
        """
        Fetch one-hop neighbors (both subject and object) and their labels.
        Returns a list of triples in the form (subject_label, predicate_label, object_label)
        """
        queryleft = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dblp: <https://dblp.org/rdf/schema#>

        SELECT DISTINCT  ?sLabel ?p  ?oLabel WHERE {{
         
            VALUES ?s {{ <{entity_uri[0]}> }}
            ?s ?p ?o .
            OPTIONAL {{ ?s rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?sLabel  }}
            OPTIONAL {{ ?p rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?pLabel  }}
            OPTIONAL {{ ?o rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?oLabel  }}
            FILTER (?p NOT IN (dblp:signatureCreator,dblp:signaturePublication,dblp:hasSignature))
        
        }}
        limit 50
        """
        queryright = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dblp: <https://dblp.org/rdf/schema#>


        SELECT DISTINCT  ?sLabel  ?pLabel  ?oLabel WHERE {{
        
            VALUES ?o {{ <{entity_uri[0]}> }}
            ?s ?p ?o .
            OPTIONAL {{ ?s rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?sLabel }}
            OPTIONAL {{ ?p rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?pLabel }}
            OPTIONAL {{ ?o rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?oLabel  }}
            FILTER (?p NOT IN (dblp:signatureCreator,dblp:signaturePublication,dblp:hasSignature))
            
        
        }}
        limit 50
        """
        params = {
        "query": queryleft,
        "action": "tsv_export"
        }
        responseleft = requests.get(self.endpoint, params=params)
        params = {
        "query": queryright,
        "action": "tsv_export"
        }
        responseright = requests.get(self.endpoint, params=params)
        return str(responseleft.text),str(responseright.text)
    
    def linearise_neighbourhood(self, left, right):
        """
        Linearizes the one-hop neighborhood into a list of strings.
        Each string is a formatted representation of the triple.
        """
        entityNeighbourhood = []
        # print(entity)
        # print("LEFT ------------")
        # print(left.replace('\t',' ').split('\n')[1:])
        # print("RIGHT ------------")
        # print(right.replace('\t',' ').split('\n')[1:])
        # print("======================")
        leftNodeNeighbourhood = [x for x in left.strip().replace('\t',' ').split('\n')[1:] if '_:bn' not in x] #no blank nodes
        rightNodeNeighbourhood = [x for x in right.strip().replace('\t',' ').split('\n')[1:] if '_:bn' not in x] #no blank nodes
        entityNeighbourhood.extend(rightNodeNeighbourhood)
        entityNeighbourhood = [x for x in entityNeighbourhood if x and x.strip()] #only triples
        return entityNeighbourhood


    def rerank_candidates(self, text, spans, entity_candidates):
        """
        Reranks the candidate entities based on their scores.
        Returns a list of tuples (entity_uri, score) sorted by score.
        """
        sorted_spans = []
        for span,entity_uris in zip(spans,entity_candidates):
            entity_scores = []
            for entity_uri in entity_uris:
                print("Fetching one-hop neighbors for entity URI...",entity_uri)
                left, right = self.fetch_one_hop(entity_uri)
                # Linearize the neighborhood
                entity_neighborhood = self.linearise_neighbourhood(left, right)
                if not entity_neighborhood:
                    print(f"No neighborhood found for entity {entity_uri[0]}")
                    continue
                # Score the entity based on its neighborhood
                print(f"Scoring entity {entity_uri[0]} with neighborhood size {len(entity_neighborhood)}")
                score,sentence = self.compute_yes_score(span['label'], text, entity_uri[0], entity_neighborhood)
                entity_scores.append((entity_uri, score, sentence))
            # Sort by score in descending order
            entity_scores.sort(key=lambda x: x[1], reverse=True)
            sorted_spans.append({'span': span, 'entities': entity_scores})
        # Return the sorted list of entity URIs and their scores    
        return sorted_spans
    

if __name__ == "__main__":
    # Example usage
    config = {
        "sparql_endpoint": "http://localhost:7015"
    }
    reranker = CandidateReranker(config)
    text = "which papers in neurips was authored by Biemann?"
    spans = [{"type": "person", "label": "Biemann"}, {"type": "venue", "label": "NeurIPS"}]
    entity_candidates = [['https://dblp.org/pid/306/6142' ,'https://dblp.org/pid/20/6100'],['https://dblp.org/streams/conf/gazeml','https://dblp.org/streams/conf/nips']] # Example URIs
    sorted_spans = reranker.rerank_candidates(text, spans, entity_candidates)
    print("Final Reranked Entities:")
    for sorted_span in sorted_spans:
        print(f"Span: {sorted_span['span']}")
        for entity_uri, score, sentence in sorted_span['entities']:
            print(f"  Entity: {entity_uri}, Score: {score:.4f} Sentence: {sentence}")
    print("Reranking completed.")