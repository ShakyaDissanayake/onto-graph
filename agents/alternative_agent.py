from .base_agent import BaseAgent
from rdflib import Graph, Namespace
from typing import List, Dict
import logging

class AlternativeAgent(BaseAgent):
    def __init__(self, rdf_graph: Graph, namespace: Namespace):
        super().__init__(rdf_graph, namespace)
        self.query_template = """
            PREFIX ns: <http://www.example.org/DrugInteraction.owl#>
            SELECT DISTINCT ?drug1 ?drug2
            WHERE {{
                ?drug1 ns:hasAlternative ?drug2 .
                FILTER(STRENDS(str(?drug1), "#{drug}") || STRENDS(str(?drug2), "#{drug}"))
            }}
        """
        logging.debug("AlternativeAgent initialized.")
    
    def get_alternatives(self, drug_name: str) -> List[Dict[str, str]]:
        query = self.query_template.format(drug=drug_name)
        logging.info(f"Fetching alternatives for {drug_name}")
        return self.query_ontology(query)
