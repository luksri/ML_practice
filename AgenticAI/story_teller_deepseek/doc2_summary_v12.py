import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_docling import DoclingLoader

from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
import pandas as pd
import faiss

from langchain_community.vectorstores import FAISS
from text_to_image_v2 import generate_infographic
from video_gen import video_gen
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
HF_TOKEN=os.environ['HF_TOKEN']

class ClinicalTrialDocument(BaseModel):
    ParticipantInformation:str=Field(None,description="Reason for participation, how participation can help.")
    Study:str=Field(None,description="goal of the study, Sponsor for the study")
    DrugUnderStudy:str=Field(None,description="Drug name, how it works, risks involved")
    TrialOverview:str=Field(None,description="The trial's objectives, treatment or intervention being tested, key outcomes being measured, medical care provided, trial period (calendar days)")
    TreatmentDosage:str=Field(None,description="The treatment regimen being tested, including dosage, frequency, and duration of treatment")
    SafetyAdverseEvents:str=Field(None,description="Any adverse events (AEs) and serious adverse events (SAEs) reported, their frequency and severity, and how they were managed")
    EfficacyResults:str=Field(None,description="Key findings on the efficacy of the treatment, including comparison to placebo or standard of care, and any statistical analysis performed")
    EthicalConsiderations:str=Field(None,description="Ethical guidelines followed, informed consent procedures, and any other ethical concerns addressed in the trial")
    StatisticalMethods:str=Field(None,description="The statistical techniques used for data analysis, including any major statistical tests and how outcomes were measured")
    ConclusionRecommendations:str=Field(None,description="The final conclusions of the trial, including whether the treatment was successful, any recommendations for future research, and any potential next steps (e.g., regulatory approval)")
    RegulatoryCompliance:str=Field(None,description="Any regulatory approvals or compliance measures mentioned, such as adherence to FDA or EMA guidelines")
    TrialPhasesMilestones:str=Field(None,description="The phases of the trial (I, II, III, IV), key milestones, and the number of participants in each phase")
    ParticipantExperience:str=Field(None,description="Any insights into the participant experience, including trial procedures, expectations, and feedback from participants (if available)")

pydantic_parser=PydanticOutputParser(pydantic_object=ClinicalTrialDocument)
format_instructions = pydantic_parser.get_format_instructions()


user_query = [
"1. Why might a participant choose to join this clinical trial? Consider motivations such as personal health improvements.",
"2. What is the primary objective of this clinical study? Clearly state whether the study is evaluating a new drug, treatment method, or specific outcomes.",
"3. Who is sponsoring this clinical trial? If missing, state: 'The document does not explicitly confirm the study sponsor.'",
"4. What is the name of the drug being tested? What is its mechanism of action?",
"5. What is the total duration of the clinical trial for an individual participant (in days)?",
"6. What potential side effects and risks are listed in the document for participants?",
"7. What specific treatment or intervention is being tested in the trial?",
"8. What medical care is provided to participants during the trial? Mention routine procedures such as check-ups, lab tests, and monitoring.",
"9. What adverse events are mentioned in the document, and how are they monitored or managed?",
"10. What ethical guidelines does the trial follow? Include references to informed consent, IRB approval, and other relevant regulations.",
"11. What statistical methods are used for data analysis? If missing, state: 'The document does not explicitly mention statistical methods.'",
"12. Does the document provide any final conclusions or recommendations based on the study? If not, mention that results are pending.",
"13. What regulatory approvals or compliance measures are specified in the document?",
"14. What are the phases of the trial, and how many participants are involved in each phase? If missing, state: 'The document does not explicitly mention trial phases.'",
"15. Are there any insights into the participant experience mentioned in the document? Consider expectations, feedback, or participant rights."
]
# user_query = """ 
# 1. Why might a participant choose to join this clinical trial? Consider motivations such as potential benefits, contribution to medical research, or personal health improvements.
# 2. What is the primary objective of this clinical study? Clearly state whether the study is evaluating a new drug, treatment method, or specific outcomes.
# 3. Who is sponsoring this clinical trial? If missing, state: "The document does not explicitly confirm the study sponsor."
# 4. What is the name of the drug being tested? What is its mechanism of action?
# 5. What is the total duration of the clinical trial for an individual participant (in days)?
# 6. What potential side effects and risks are listed in the document for participants?
# 7. What specific treatment or intervention is being tested in the trial?
# 8. What medical care is provided to participants during the trial? Mention routine procedures such as check-ups, lab tests, and monitoring.
# 9. What adverse events are mentioned in the document, and how are they monitored or managed?
# 10. What ethical guidelines does the trial follow? Include references to informed consent, IRB approval, and other relevant regulations.
# 11. What statistical methods are used for data analysis? If missing, state: "The document does not explicitly mention statistical methods."
# 12. Does the document provide any final conclusions or recommendations based on the study? If not, mention that results are pending.
# 13. What regulatory approvals or compliance measures are specified in the document?
# 14. What are the phases of the trial, and how many participants are involved in each phase? If missing, state: "The document does not explicitly mention trial phases."
# 15. Are there any insights into the participant experience mentioned in the document? Consider expectations, feedback, or participant rights.

# """
# ]
# PROMPT_TEMPLATE = PromptTemplate.from_template("""
#     You are an expert in clinical trials and medical research. 
#     - Your task is to analyze the provided document and answer user queries in a formal, concise, and informative manner.
#     - Use the provided document context to answer the questions.
#     - Limit responses to two to three sentences per question.
#     - If a specific detail is missing, state: "The document does not explicitly mention this information.
#     - If the document does not explicitly mention a detail, state that the information is not available.
#     - Do not infer or fabricate information beyond what is provided in the document.
#     - Maintain clarity, accuracy, and a patient-friendly tone when necessary.
    
#     Question: {user_query}
#     Context: {document_context}
#     Answer: 

# """,
# partial_variables={"format_instructions": format_instructions})


PDF_STORAGE_PATH = './document_store/pdfs/'

EMBED_MODEL_ID = "nomic-ai/nomic-embed-text-v1"
EMBEDDING_MODEL = OllamaEmbeddings(model="nomic-embed-text")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="llama3.2:1b", params={"temperature": 0.01, "seed": 42, "top_k": 0,"top_p":0.9})


def load_doc_documents(file_path):
    # document_loader = Docx2txtLoader(file_path)
    document_loader = DoclingLoader(file_path,export_type=ExportType.DOC_CHUNKS,chunker=HybridChunker(tokenizer=EMBED_MODEL_ID,
                                                                                                      max_tokens=2048,
                                                                                                      token=HF_TOKEN))
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)


def get_embedding(docs):
    """Generates an embedding for the given text."""
    #documents=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(docs)
    vectordb=FAISS.from_documents(docs,EMBEDDING_MODEL)
    # print(vectordb.index.ntotal)
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3} )
    # retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.75} )
    return retriever

def summarize_with_llm(text):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""
    global vstore

    retrieved_text=vstore.invoke(text)
    document_text=[i.page_content for i in retrieved_text]
    document_text="/n".join(document_text)
    
    prompt = f"""
        You are an expert in clinical trials and medical research. 
        - Your task is to analyze the provided document and answer user queries in a formal, concise, and informative manner.
        - Limit responses to two to three sentences per question.
        - Do not infer or fabricate information beyond what is provided in the document.
        - you must address the individual as a 'participant.'
      
        User Interaction :
            Once the clinical trial context is processed, you should respond accurately to user queries by referencing the extracted information. 
            If a query is outside the provided context, state so clearly.
                                        
        Tone & Style Guidelines:
            Maintain clarity, accuracy, and a patient-friendly tone when necessary.

        Question: {text}
        Context: {document_text}
        Answer: 

    """

    response = LANGUAGE_MODEL.invoke(prompt)
    
    return response


raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
processed_chunks = chunk_documents(raw_docs)
vstore = get_embedding(raw_docs)
# summary = summarize_with_llm(user_query)
# print(summary)

for ques in user_query:
    print(f"----- User question : {ques} -----")
    summary = summarize_with_llm(ques)
    print(summary)