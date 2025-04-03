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
import re
from langchain_community.vectorstores import FAISS
from text_to_image_v2 import generate_infographic
from video_gen import video_gen
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import openai
import json
import rich

load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv("OPENAI_API_KEY")
OPENAI_API_KEY=os.environ['OPENAI_API_KEY']


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

def load_doc_documents(file_path):
    document_loader = DoclingLoader(file_path,export_type=ExportType.MARKDOWN)
    return document_loader.load()


def summarize_with_llm(text):

    system_message = f"""
        You are an expert in medical context understanding with deep knowledge of clinical trials. Your task is to extract and summarize all relevant and granular details from the given clinical trial document context passed to you, while maintaining precision, depth, and completeness.

        Instructions:
        Retrieve all key facts, numerical data, contextual insights, implicit relationships, metadata, and supporting references from the context. Ensure no critical information is omitted, even if seemingly minor. However, limit finer details regarding dosing and schedules.

        Summarize the clinical trial context in a structured format, covering the following aspects:
            - **Participant Information**: Reason for participation, how participation can help.
            - **Study**: goal of the study, Sponsor for the study. 
            - **Drug Under Study**: Drug name, how it works, risks involved.
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

        Tone: 
            - Present the summary with a structured and professional tone, suitable for medical research and regulatory review.
            Output under each section should be clear, detailed, and organized, ensuring a high level of accuracy in medical document analysis.

        Only return the output in the following format : {format_instructions}, avoid any other information in the output.
        """

    # OpenAI API call using the latest SDK format
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
       model="gpt-4o",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": text[0].page_content},
        ],
        temperature=0.7,
        max_tokens=800
    )
    summary = response.choices[0].message.content
    return summary


raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
summary = summarize_with_llm(raw_docs)    
parsed_output = pydantic_parser.parse(summary)
rich.print(parsed_output.model_dump())

image_num = 1
video_dict = dict()
for text, summ in parsed_output.model_dump().items():
    prompt = f"A colorful, cartoon-style infographic illustrating {summ}. The image should be visually engaging, using simple characters, medical symbols, and a friendly, informative tone."
    image_name = f"./image_gen/{image_num}.png"
    generate_infographic(prompt, image_name)
    video_dict[summ]=image_name 
    image_num += 1

if video_dict:
    video_gen(list(video_dict.keys()), list(video_dict.values()), video_name="stroy_v16_openai_1")
else:
    print(f"dicts are empty: {len(video_dict), {len(video_dict)}}")
