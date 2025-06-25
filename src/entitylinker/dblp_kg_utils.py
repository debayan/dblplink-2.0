import json
import requests
from typing import List, Tuple
from urllib.parse import urlencode
import sys


class OneHopFetcher:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.endpoint = config["sparql_endpoint"]
        self.headers = {
            "Accept": "application/sparql-results+json"
        }

    def fetch_one_hop(self, entity_uri: str) -> List[Tuple[str, str, str]]:
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

        SELECT DISTINCT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
         
            VALUES ?s {{ <{entity_uri}> }}
            ?s ?p ?o .
            OPTIONAL {{ ?s rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?sLabel  }}
            OPTIONAL {{ ?p rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?pLabel  }}
            OPTIONAL {{ ?o rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?oLabel  }}
            FILTER (?p NOT IN (dblp:signatureCreator,dblp:signaturePublication,dblp:hasSignature))
        
        }}
        limit 100
        """
        queryright = f"""
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
        PREFIX dc: <http://purl.org/dc/elements/1.1/>
        PREFIX foaf: <http://xmlns.com/foaf/0.1/>
        PREFIX dblp: <https://dblp.org/rdf/schema#>


        SELECT DISTINCT ?s ?sLabel ?p ?pLabel ?o ?oLabel WHERE {{
        
            VALUES ?o {{ <{entity_uri}> }}
            ?s ?p ?o .
            OPTIONAL {{ ?s rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?sLabel }}
            OPTIONAL {{ ?p rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?pLabel }}
            OPTIONAL {{ ?o rdfs:label|skos:prefLabel|dc:title|foaf:name|dblp:abstract|dc:description|dblp:title ?oLabel  }}
            FILTER (?p NOT IN (dblp:signatureCreator,dblp:signaturePublication,dblp:hasSignature))
            
        
        }}
        limit 100
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


# Example usage:
if __name__ == "__main__":
    config_path = "config.json"
    entities = ["https://dblp.org/pid/213/7475","https://dblp.org/rec/conf/desrist/PoserWBSPB22"]
    for entity in entities:
        fetcher = OneHopFetcher(config_path)
        left,right = fetcher.fetch_one_hop(entity)
        for sentence in left.split('\n')[1:]:
            print("left",sentence)
        for sentence in right.split('\n')[1:]:
            print(sentence,"right")
