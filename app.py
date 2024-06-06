import streamlit as st
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
from analyzers import DrugInteractionAnalyzer
import base64

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
    normalized_input: Optional[str] = None
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

def normalize_input_node(state: GraphState) -> Dict[str, str]:
    """Normalize drug names using LLM to correct spelling and case."""
    valid_drugs = [
        "Warfarin", "Aspirin", "Simvastatin", "Erythromycin", "Metformin",
        "Ciprofloxacin", "Lisinopril", "Spironolactone", "Clopidogrel",
        "Omeprazole", "Apixaban", "Rosuvastatin", "Empagliflozin", "Losartan",
        "Pantoprazole", "Metronidazole", "Theophylline", "Atorvastatin",
        "Phenelzine", "Ibuprofen", "Naproxen", "Amlodipine", "Nifedipine",
        "Esomeprazole", "Glucophage", "Sertraline", "Fluoxetine"
    ]
    
    template = """
    You are a pharmaceutical expert. Your task is to correct any spelling mistakes and ensure proper 
    capitalization of the following drug names (separated by commas). You must only suggest drugs from 
    the following valid list:

    Valid drugs: {valid_drugs}

    Original drug names: {input}

    Rules:
    1. Only suggest drugs from the valid list
    2. If a drug is not in the valid list, respond with "Drug not in the Ontology because this is a Demo version"
    3. For drugs like "Glucophage", also recognize it as "Metformin" as they are the same medication
    4. Provide the corrected names in a comma-separated format
    5. If none of the input drugs match or are similar to the valid list, respond with "None of the provided drugs are available in this demo version"

    Please provide the corrected drug names in a comma-separated format only, without any additional text.
    
    Examples:
    Input: "ciprofloxacin, lisinopril"
    Output: "Ciprofloxacin, Lisinopril"
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    try:
        llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=GROQ_API_KEY,
            temperature=0.1  # Lower temperature for more consistent results
        )
        
        chain = prompt | llm
        response = chain.invoke({
            "input": state.input,
            "valid_drugs": ", ".join(valid_drugs)
        })
        normalized_input = response.content.strip()
        
        # Remove any quotes that might be in the response
        normalized_input = normalized_input.replace('"', '')
        
        # Handle invalid responses with more user-friendly messages
        if normalized_input in ["Drug not in the Ontology because this is a Demo version", "None of the provided drugs are available in this demo version"]:
            logging.warning(f"Drugs not in demo ontology: {state.input}")
            return {"normalized_input": "The specified medication(s) are not available in this demo version. Please try medications from the ontology."}
        
        # Verify that all normalized drugs are in the valid list
        normalized_drugs = [drug.strip() for drug in normalized_input.split(",")]
        
        # Debug logging
        logging.info(f"Normalized drugs: {normalized_drugs}")
        logging.info(f"Valid drugs: {valid_drugs}")
        
        valid_check = all(drug in valid_drugs for drug in normalized_drugs)
        logging.info(f"Valid check result: {valid_check}")
        
        if not valid_check:
            invalid_drugs = [drug for drug in normalized_drugs if drug not in valid_drugs]
            logging.warning(f"Invalid drugs found: {invalid_drugs}")
            return {"normalized_input": "Some of the specified medications are not available in this demo version. Please use medications from the valid list shown above."}
        
        logging.info(f"Normalized drug names: {normalized_input}")
        return {"normalized_input": normalized_input}
        
    except Exception as e:
        logging.error(f"Error in drug name normalization: {e}")
        return {"normalized_input": state.input}  # Fall back to original input

def create_agent_graph(owl_file_path: str) -> StateGraph:
    """Create a processing graph for drug interaction analysis using separate agents."""
    analyzer = DrugInteractionAnalyzer(owl_file_path)
    
    def user_input_node(state: GraphState) -> Dict[str, str]:
        logging.info("Processing normalized user input.")
        return {"query": state.normalized_input}
    
    def ontology_query_node(state: GraphState) -> Dict[str, str]:
        try:
            logging.info("Executing ontology queries.")
            drug_names = [d.strip() for d in state.normalized_input.split(",")]
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
        
        Original drug names: {input}
        Normalized drug names: {normalized_input}
        
        Please provide a comprehensive summary of:
        1. Direct interactions between the drugs
        2. Potential conflicts
        3. Similar drug alternatives
        4. Recommended alternatives if conflicts exist
        
        If no results were found, please indicate this clearly.
        Format the response in a clear, structured manner.
        Also mention if any drug names were corrected during normalization.
        """
        
        prompt = ChatPromptTemplate.from_template(template)

        try:
            llm = ChatGroq(
                model_name="llama-3.3-70b-versatile",
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
                "ontology_results": state.ontology_results,
                "input": state.input,
                "normalized_input": state.normalized_input
            })

            logging.info("LLM processing completed successfully.")
            return {"response": response.content}
        except Exception as e:
            logging.error(f"Error processing results with LLM: {e}")
            return {"response": f"Error processing results: {str(e)}"}

    # Create and configure the graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("normalize_input", normalize_input_node)
    workflow.add_node("input_processor", user_input_node)
    workflow.add_node("ontology_query", ontology_query_node)
    workflow.add_node("llm_processing", llm_processing_node)

    # Add edges
    workflow.add_edge("normalize_input", "input_processor")
    workflow.add_edge("input_processor", "ontology_query")
    workflow.add_edge("ontology_query", "llm_processing")

    # Set entry point
    workflow.set_entry_point("normalize_input")

    logging.info("Agent graph created and configured successfully.")

    return workflow.compile()

def main():
    st.set_page_config(page_title="OntoGraph", page_icon="💊", layout="wide")

    st.markdown("<h1 style='text-align: center;'>OntoGraph - Drug Interaction Analysis System</h1>", unsafe_allow_html=True)
    
    st.markdown("<h3 style='text-align: center;'>{ 💊 ↔ 💊 = ❓}</h3><br>", unsafe_allow_html=True)
    
    st.markdown("<h5 style='text-align: center; color: #8bacad;'>This application uses a combination of ontology-based reasoning and language models to analyze drug interactions.</h5>", unsafe_allow_html=True)
        
    st.markdown("""
                <style>
                .centered {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }
                </style>
                <div class="centered">
            """, unsafe_allow_html=True)

    def get_base64_image(img_path):
        with open(img_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f"data:image/webp;base64,{encoded}"

    # Generate the image URL
    image_url = get_base64_image("img/header-image.webp")

    # Load custom CSS with the embedded image
    st.markdown(f"""
        <style>
            .parallax {{
                background-image: url("{image_url}");
                min-height: 200px;
                background-attachment: fixed;
                background-position: center;
                background-repeat: no-repeat;
                background-size: cover;
                opacity: 0.6;
            }}
        </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="parallax"></div>', unsafe_allow_html=True) 
    
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([3,7])

    with col1:
        # Features
        st.markdown("<h3 style='text-align: left; color: #5ac5c9;'>Features</h3>", unsafe_allow_html=True)
        st.write("""
        - Drug interaction detection
        - Conflict identification
        - Similar drug suggestions
        - Alternative medication recommendations
        """)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Instructions
        st.markdown("<h3 style='text-align: left; color: #d46c6c;'>Instructions</h3>", unsafe_allow_html=True)
        st.write("""
        1.   Enter the drug names separated by commas.
        2.   Click on the 'Analyze' button.
        3.   Wait for the analysis to complete.
        4.   View the results below.
        """)

    with col2:
        # Input
        st.markdown("""`Supported Drugs for this Demo Version` : Warfarin, Aspirin, Simvastatin, Erythromycin, Metformin,
                    Ciprofloxacin, Lisinopril, Spironolactone, Clopidogrel,
                    Omeprazole, Apixaban, Rosuvastatin, Empagliflozin, Losartan,
                    Pantoprazole, Metronidazole, Theophylline, Atorvastatin,
                    Phenelzine, Ibuprofen, Naproxen, Amlodipine, Nifedipine,
                    Esomeprazole, Glucophage, Sertraline, Fluoxetine""")
        st.markdown('<h3 class="big-font">Enter drug names separated by commas:</h3>', unsafe_allow_html=True)
        user_input = st.text_input("", value="", key="drug_input", placeholder="e.g., Aspirin, Warfarin")
        st.markdown('<style>div[data-testid="stTextInput"] input { font-size: 16px;}</style>', unsafe_allow_html=True)
        
        if st.button("Analyze", type="primary"):
            if not user_input.strip():
                st.warning("Please enter at least one drug name.")
                return

            owl_file_path = os.path.join("ontology", "DrugInteraction.owl")
            if not os.path.exists(owl_file_path):
                logging.error(f"Ontology file not found: {owl_file_path}")
                st.error(f"Ontology file not found: {owl_file_path}")
                return

            try:
                with st.spinner("Analyzing drug interactions..."):
                    agent_graph = create_agent_graph(owl_file_path)
                    result = agent_graph.invoke(GraphState(input=user_input))
                    
                    # Display normalized input if it's different from original
                    if result.get("normalized_input") and result["normalized_input"] != user_input:
                        st.info(f"Normalized drug names: {result['normalized_input']}")

                st.subheader("Analysis Results:")
                st.markdown(result["response"])

                logging.info("Analysis completed and results displayed.")
            except Exception as e:
                logging.error(f"An error occurred: {str(e)}")
                st.error(f"An error occurred: {str(e)}")

        st.markdown("<hr>", unsafe_allow_html=True)
        
        # Disclaimer
        st.write("""
        <div style='text-align: center;'>
            <p style='text-align: left; color: #d46c6c;'>Disclaimer: </p><p style='text-align: left; color: #707377;'>This application is intended for informational purposes only and does not replace professional medical advice, diagnosis, or treatment. The analysis provided is based on the data available in the ontology and may not account for all possible drug interactions. Users are strongly advised to consult a licensed healthcare provider before making any decisions based on the analysis results. The creators of this application are not responsible for any decisions made or actions taken based on the information provided.</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()