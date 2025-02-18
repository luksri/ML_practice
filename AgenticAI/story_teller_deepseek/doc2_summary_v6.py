"""
dont use any vector db. upload the doc and pass it as context
"""
import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from text_to_image_v2 import generate_infographic


PDF_STORAGE_PATH = './document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0.4, "seed": 42, "top_k": 1})


def load_doc_documents(file_path):
    document_loader = DoclingLoader(file_path)
    docs = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


def summarize_with_llm(text):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""
    
    prompt = f"""
    Please summarize the following clinical trial document in detail using the given context, covering the following aspects:  
    - **Participant Information**: Reason for participation, how participation can help.
    - **Study Goal** - If missing, state: "The document does not explicitly mention the study goal."
    - **Sponsor** - If missing, state: "The document does not explicitly confirm the study sponsor."
    - **Drug Under Study** - Name, function and work.
    - **Trial Duration** - Number of days.
    - **Side Effects** - Risks and impacts on patient health.
    - **Trial Overview**: The trial's objectives, treatment or intervention being tested, key outcomes being measured, medical care provided, trial period (calendar days).  
    - **Treatment and Dosage**: The treatment regimen being tested, including dosage, frequency, and duration of treatment.  
    - **Safety and Adverse Events**: Any adverse events (AEs) and serious adverse events (SAEs) reported, their frequency and severity, and how they were managed.  
    - **Efficacy Results**: Key findings on the efficacy of the treatment, including comparison to placebo or standard of care, and any statistical analysis performed.  
    - **Ethical Considerations**: Ethical guidelines followed, informed consent procedures, and any other ethical concerns addressed in the trial.  
    - **Statistical Methods**: The statistical techniques used for data analysis, including any major statistical tests and how outcomes were measured.  
    - **Conclusion and Recommendations**: The final conclusions of the trial, including whether the treatment was successful, any recommendations for future research, and any potential next steps (e.g., regulatory approval).  
    - **Regulatory Compliance**: Any regulatory approvals or compliance measures mentioned, such as adherence to FDA or EMA guidelines.  
    - **Trial Phases and Milestones**: The phases of the trial (I, II, III, IV), key milestones, and the number of participants in each phase.  
    - **Patient/Participant Experience**: Any insights into the participant experience, including trial procedures, expectations, and feedback from participants (if available).

    
    Context: {text}
    """
    # print(retrieved_text)
    response = LANGUAGE_MODEL.invoke(prompt)
    
    return response

raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
ai_response = summarize_with_llm(raw_docs)
print(ai_response)
