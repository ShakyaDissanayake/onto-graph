
from rdflib import Graph, Namespace, URIRef, Literal
from typing import Dict, List, Optional
from langgraph.graph import StateGraph
from langchain.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from langchain_community.chat_models import ChatOllama
from langchain_groq import ChatGroq
import logging
# Import the DrugInteractionAnalyzer
from analyzers import DrugInteractionAnalyzer


# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Validating API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")
    raise ValueError("GROQ_API_KEY not found in environment variables. Please add it to your .env file.")

@dataclass
class GraphState:
    """State type for the graph."""
    input: str
    query: Optional[str] = None
    ontology_results: Optional[str] = None
    response: Optional[str] = None

class OntologyAgent:
    def __init__(self, owl_file_path: str):
        """Initialize the OntologyAgent with an OWL file."""

        self.g = Graph()
        try:
            self.g.parse(owl_file_path, format="xml")
            self.ns = Namespace("http://www.example.org/DrugInteraction.owl#")
            logging.info(f"Ontology loaded successfully from {owl_file_path}")
        except Exception as e:
            logging.error(f"Failed to load ontology file: {e}")
            raise ValueError(f"Failed to load ontology file: {e}")

def create_agent_graph(owl_file_path: str) -> StateGraph:
    """Create a processing graph for drug interaction analysis using separate agents."""
    analyzer = DrugInteractionAnalyzer(owl_file_path)
    
    def user_input_node(state: GraphState) -> Dict[str, str]:
        logging.info("Processing user input.")

        return {"query": state.input}
    
    def ontology_query_node(state: GraphState) -> Dict[str, str]:
        try:
            logging.info("Executing ontology queries.")
            drug_names = [d.strip() for d in state.input.split(",")]
            results = analyzer.analyze_drugs(drug_names)
            logging.info(f"Ontology query results: {results}")
            return {"ontology_results": json.dumps(results, indent=2)}
        except Exception as e:
            logging.warning(f"Ontology query failed: {e}")

            return {"ontology_results": json.dumps({"error": str(e)})}
    
    def llm_processing_node(state: GraphState) -> Dict[str, str]:
        template = """
        Based on the drug interaction analysis results:
        {ontology_results}
        
        Please provide a comprehensive summary of:
        1. Direct interactions between the drugs
        2. Potential conflicts
        3. Similar drug alternatives
        4. Recommended alternatives if conflicts exist
        
        If no results were found, please indicate this clearly.
        Format the response in a clear, structured manner.
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        try:
            llm = ChatGroq(
                model_name="llama3-groq-70b-8192-tool-use-preview",
                api_key=GROQ_API_KEY,
                temperature=0.7
            )
            logging.info("LLM initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            return {"response": f"Error initializing LLM: {str(e)}"}

        chain = prompt | llm

        try:
            response = chain.invoke({
                "ontology_results": state.ontology_results
            })

            logging.info("LLM processing completed successfully.")
            return {"response": response.content}
        except Exception as e:
            logging.error(f"Error processing results with LLM: {e}")
            return {"response": f"Error processing results: {str(e)}"}

    # Create and configure the graph
    workflow = StateGraph(GraphState)

    workflow.add_node("input_processor", user_input_node)
    workflow.add_node("ontology_query", ontology_query_node)
    workflow.add_node("llm_processing", llm_processing_node)

    workflow.add_edge("input_processor", "ontology_query")
    workflow.add_edge("ontology_query", "llm_processing")

    workflow.set_entry_point("input_processor")

    logging.info("Agent graph created and configured successfully.")

    return workflow.compile()

def main():
    """Main function to run the drug interaction analysis."""
    try:
        logging.info("Starting Drug Interaction Analysis System.")
        

        print("Drug Interaction Analysis System")
        print("Enter drug names separated by commas (e.g., Aspirin, Warfarin):")
        user_input = input("Drugs: ").strip()
        
        if not user_input:
            logging.warning("No drug names provided. Exiting.")
            print("No drug names provided. Exiting.")
            return
        
        owl_file_path = os.path.join("ontology", "DrugInteraction.owl")
        if not os.path.exists(owl_file_path):
            logging.error(f"Ontology file not found: {owl_file_path}")

            raise FileNotFoundError(f"Ontology file not found: {owl_file_path}")
        
        agent_graph = create_agent_graph(owl_file_path)
        result = agent_graph.invoke(GraphState(input=user_input))
        
        print("\nAnalysis Results:")
        print(result["response"])

        logging.info("Analysis completed and results displayed.")
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        print(f"An error occurred: {str(e)}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()

