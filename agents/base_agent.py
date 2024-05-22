from rdflib import Graph, Namespace, URIRef
from typing import List, Dict
import logging

class BaseAgent:
    def __init__(self, rdf_graph: Graph, namespace: Namespace):
        self.graph = rdf_graph
        self.ns = namespace
        logging.debug("BaseAgent initialized.")
    
    def query_ontology(self, query: str) -> List[Dict[str, str]]:
        """Execute a SPARQL query and return results as a list of dictionaries."""
        try:
            results = self.graph.query(query)
            logging.debug(f"Executing query: {query}")
            return [
                {"drug1": str(row[0]).split('#')[-1], 
                 "drug2": str(row[1]).split('#')[-1]} 
                for row in results
            ]
        except Exception as e:
            logging.error(f"Error executing query: {e}")
            return []
