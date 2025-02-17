import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer
from docx import Document
from langchain_community.vectorstores import FAISS
from langchain_docling import DoclingLoader
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings

embeddings=(
    OllamaEmbeddings(model="llama2:13b")  ##by default it ues llama2
)

LANGUAGE_MODEL = OllamaLLM(model="llama2:13b", params={"temperature": 0, "seed": 42, "top_k": 1})
user_query = """
- participant asked to participate in the clinical research because?
- what is the goal of the study?
- what is the drug under study?
- explain how drug works?
- explain side effects and impacts w.r.to patient health?
- what are the risks involved?
- who is the sponsor for the study?
- what is the medicalcare provided?
- what is the trial period? (should be number of calendar days)
- (optional) do you have any statistics for the number of participants participating?
"""

user_prompt="""
Please summarize the following clinical trial document in detail, covering the following aspects:

    - Trial Overview: The trial's objectives, design (randomized, placebo-controlled, etc.), treatment or intervention being tested, and key outcomes being measured.
    - Participant Information: Inclusion and exclusion criteria, participant demographics (age, gender, health status), and sample size.
    - Treatment and Dosage: The treatment regimen being tested, including dosage, frequency, and duration of treatment.
    - Safety and Adverse Events: Any adverse events (AEs) and serious adverse events (SAEs) reported, their frequency and severity, and how they were managed.
    - Efficacy Results: Key findings on the efficacy of the treatment, including comparison to placebo or standard of care, and any statistical analysis performed.
    - Ethical Considerations: Ethical guidelines followed, informed consent procedures, and any other ethical concerns addressed in the trial.
    - Statistical Methods: The statistical techniques used for data analysis, including any major statistical tests and how outcomes were measured.
    - Conclusion and Recommendations: The final conclusions of the trial, including whether the treatment was successful, any recommendations for future research, and any potential next steps (e.g., regulatory approval).
    - Regulatory Compliance: Any regulatory approvals or compliance measures mentioned, such as adherence to FDA or EMA guidelines.
    - Trial Phases and Milestones: The phases of the trial (I, II, III, IV), key milestones, and the number of participants in each phase.
    - Patient/Participant Experience: Any insights into the participant experience, including trial procedures, expectations, and feedback from participants (if available).
"""

def load_doc_documents(file_path):
    # document_loader = Docx2txtLoader(file_path)
    document_loader = DoclingLoader(file_path)
    return document_loader.load()

def get_embedding(docs):
    """Generates an embedding for the given text."""
    documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    vectordb=FAISS.from_documents(documents,embeddings)
    retriever=vectordb.as_retriever()
    return retriever
     

def summarize_with_llm(text, vstore):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""

    retrieved_text=vstore.invoke(text)
    
    prompt = f"""
    create a short summary with the given context for all the user queries {user_query}.
    - you should serve to highlight important points that may be deciding factor for potential participants who may or may not want to join.
    - Always refer to the individual as a 'participant.'
    - If the information is not available, state that you don't know.
    - Optional details can be omitted if there isn't enough information.
    
    context: {retrieved_text}
    """
    
    response = LANGUAGE_MODEL.invoke(prompt)
    
    return response
    

# Example Usage
docx_path = "./HRP-503 - SAMPLE Biomedical Protocol.docx"  # Replace with actual file path
text_data = load_doc_documents(docx_path)

# Initialize FAISS index (dimensionality must match embedding size, e.g., 384 for MiniLM)
vstore = get_embedding(text_data)
summary = summarize_with_llm(f"Ppts, findings, and supporting details.", vstore)
print(summary)

# for each_q in user_query:
#     summary = summarize_with_llm(each_q, vstore)
#     print(summary)

