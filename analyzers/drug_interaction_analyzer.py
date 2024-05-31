from typing import Dict, List
import json
from agents import AlternativeAgent, SimilarityAgent, ConflictAgent, InteractionAgent
from rdflib import Graph, Namespace
import logging

class DrugInteractionAnalyzer:
    """
    Analyzer class to handle drug interactions, alternatives, similarities, and conflicts using separate agents.
    """
    def __init__(self, owl_file_path: str):
        """
        Initialize the DrugInteractionAnalyzer with the ontology file and instantiate agents.
        
        Args:
            owl_file_path (str): Path to the RDF/XML ontology file.
        """
        self.graph = Graph()
        try:
            self.graph.parse(owl_file_path, format="xml")
            self.namespace = Namespace("http://www.example.org/DrugInteraction.owl#")
            logging.info(f"Successfully loaded ontology from {owl_file_path}")
        except Exception as e:
            logging.error(f"Failed to load ontology file: {e}")
            raise ValueError(f"Failed to load ontology file: {e}")
        
        # Initialize agents
        self.alternative_agent = AlternativeAgent(self.graph, self.namespace)
        self.similarity_agent = SimilarityAgent(self.graph, self.namespace)
        self.conflict_agent = ConflictAgent(self.graph, self.namespace)
        self.interaction_agent = InteractionAgent(self.graph, self.namespace)
        logging.info("Initialized all agents successfully.")
    
    def analyze_drugs(self, drug_names: List[str]) -> Dict[str, Dict[str, List[Dict[str, str]]]]:
        """
        Analyze multiple drugs using separate agents and return structured results.
        
        Args:
            drug_names (List[str]): List of drug names to analyze.
        
        Returns:
            Dict[str, Dict[str, List[Dict[str, str]]]]: Structured analysis results.
        """
        results = {}
        for drug in drug_names:
            drug_results = {}
            logging.info(f"Analyzing drug: {drug}")
            
            # Alternatives
            try:
                alternatives = self.alternative_agent.get_alternatives(drug)
                drug_results["alternatives"] = alternatives
                logging.debug(f"Alternatives for {drug}: {alternatives}")
            except Exception as e:
                logging.warning(f"Failed to get alternatives for {drug}: {e}")
                drug_results["alternatives"] = []
            
            # Similarities
            try:
                similarities = self.similarity_agent.get_similarities(drug)
                drug_results["similarities"] = similarities
                logging.debug(f"Similarities for {drug}: {similarities}")
            except Exception as e:
                logging.warning(f"Failed to get similarities for {drug}: {e}")
                drug_results["similarities"] = []
            
            # Conflicts
            try:
                conflicts = self.conflict_agent.get_conflicts(drug)
                drug_results["conflicts"] = conflicts
                logging.debug(f"Conflicts for {drug}: {conflicts}")
            except Exception as e:
                logging.warning(f"Failed to get conflicts for {drug}: {e}")
                drug_results["conflicts"] = []
            
            # Interactions
            try:
                interactions = self.interaction_agent.get_interactions(drug)
                drug_results["interactions"] = interactions
                logging.debug(f"Interactions for {drug}: {interactions}")
            except Exception as e:
                logging.warning(f"Failed to get interactions for {drug}: {e}")
                drug_results["interactions"] = []
            
            results[drug] = drug_results
            logging.info(f"Completed analysis for {drug}")
        
        logging.info("Completed analysis for all drugs.")
        return results
