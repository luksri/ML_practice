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


user_query = """ 
1. Participant is considered to take part of study because of which disease condtion?
2. What is the goal of this clinical study?
3. Who is sponsoring this clinical trial? If missing, state: "The document does not explicitly confirm the study sponsor."
4. What is the investigative drung? how it works?
5. what is the duration of the screening phase? 
6. What potential side effects and risks because of clinical trial?
7. What specific treatment or intervention is being tested in the trial?
8. What medical care is provided to participants during the trial?
9. What adverse events are mentioned in the document, and how are they monitored or managed?
10. What ethical guidelines does the trial follow? 
11. What statistical methods are used for data analysis? 

"""


EMBED_MODEL_ID = "deepseek-ai/DeepSeek-R1"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="llama3.1", params={"temperature": 0.01, "seed": 42, "top_k": 1})


def load_doc_documents(file_path):
    # document_loader = Docx2txtLoader(file_path)
    document_loader = DoclingLoader(file_path,export_type=ExportType.DOC_CHUNKS,
                                    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID,max_tokens=2048,token=HF_TOKEN))
    return document_loader.load()



def get_embedding(docs):
    """Generates an embedding for the given text."""
    vectordb=FAISS.from_documents(docs,EMBEDDING_MODEL)
    print(f"Size of vector DB : {vectordb.index.ntotal}")
    retriever=vectordb.as_retriever()
    return retriever

def summarize_with_llm(text):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""
    global vstore

    retrieved_text=vstore.invoke(text)
    document_text=[i.page_content for i in retrieved_text]
    document_text="/n".join(document_text)
    
    prompt = f"""
    You are an expert in medical context understanding with deep knowledge of clinical trials.Your task is to respond to the questions asked by user from the given clinical trial document context passed to you.

    Get a clear understanding of the context with respect to below topics, this will help you in answering the user asked questions:
        - Participant Information: Reason for participation, expected benefits, and how participation contributes to research.
        - Study Overview: Purpose and objectives of the study, along with the sponsoring organization.
        - Drug Under Study: Drug name, mechanism of action, and associated risks.
        - Trial Overview: Objectives, treatment or intervention being tested, key outcome measures, medical care provided, and trial duration (in calendar days).
        - Treatment and Dosage: Description of the treatment regimen, including dosage, frequency, and duration.
        - Safety and Adverse Events: Summary of reported adverse events (AEs) and serious adverse events (SAEs), including their frequency, severity, and management.
        - Efficacy Results: Key findings on treatment efficacy, comparison to placebo or standard of care, and statistical analysis outcomes.
        - Ethical Considerations: Ethical guidelines followed, informed consent procedures, and any addressed ethical concerns.
        - Statistical Methods: Analytical techniques used, including major statistical tests and outcome measurement approaches.
        - Conclusion and Recommendations: Final conclusions of the trial, effectiveness of the treatment, recommendations for future research, and potential next steps (e.g., regulatory approval).
        - Regulatory Compliance: Any regulatory approvals or compliance measures (e.g., FDA or EMA adherence).
        - Trial Phases and Milestones: Breakdown of trial phases (I, II, III, IV), key milestones, and participant numbers in each phase.
        - Patient/Participant Experience: Insights into participant experiences, including trial procedures, expectations, and feedback (if available).
    
    Limit response to exactly 2 sentences. Do not mention section numbers or document structure.
    Avoid excessive explanation—keep it short and fact-driven.
    Always refer to the individual as a 'participant' in response.
    Tone should be:
        - Simplifying the language: Avoid overly technical wording unless necessary.
        - Speaking directly to the reader: Use "you" and "we" when appropriate.
        - Using natural phrasing: Write how you would explain it to a friend.
        - Breaking up long sentences: Shorter sentences feel more approachable.


    User Interaction :
        Once the clinical trial context is processed, you should respond accurately to user queries. If a query is outside the provided context, state so clearly.
                                        
    context: {document_text}
    Question : {text}
    """
    response = LANGUAGE_MODEL.invoke(prompt)    
    return response

def parse_answers(response_text):
    # Split answers using the numbered format "1.", "2.", etc.
    answers = {}
    split_responses = response_text.split("\n")
    
    current_question = None
    for line in split_responses:
        line = line.strip()
        if line and line[0].isdigit() and "." in line[:3]:  # Check for question numbering
            current_question = line  # Store the question as the key
            answers[current_question] = ""
        elif current_question:  # Append text to the current question's answer
            answers[current_question] += line + " "

    return answers


raw_docs = load_doc_documents('./ex.docx')
vstore = get_embedding(raw_docs)


# summary = summarize_with_llm(user_query)    
# final_answer = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
# print(final_answer)
question_dict = dict()
for ques in user_query.split("\n"):
    print(f"------ User question : {ques} ------")
    summary = summarize_with_llm(ques)    
    final_answer = re.sub(r'<think>.*?</think>', '', summary, flags=re.DOTALL).strip()
    print(final_answer)
    if ques and ques != ' ':
        question_dict[ques] = final_answer
    print("######################End of Answer###################################")

# parsed_answers = parse_answers(final_answer)
print(question_dict)

image_num = 1
video_dict = dict()
for question, answer in question_dict.items():
    print(f"{question}\n{answer.strip()}\n")
    prompt = f"create a cartoon image for {question}"
    image_name = f"./image_gen/{image_num}.png"
    # generate_infographic(prompt, image_name)
    image_num += 1
    if answer:
        video_dict[question]=image_name

if video_dict and question_dict:
    video_gen(list(question_dict.values()), list(video_dict.values()), video_name="stroy_v14_1")
else:
    print(f"dicts are empty: {len(video_dict), {len(question_dict)}}")
