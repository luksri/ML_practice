import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
import pandas as pd

from text_to_image_v2 import generate_infographic
from video_gen import video_gen

user_query = """ 
1. Why might a participant choose to join this clinical trial? Consider motivations such as potential benefits, contribution to medical research, or personal health improvements.
2. What is the primary objective of this clinical study? Clearly state whether the study is evaluating a new drug, treatment method, or specific outcomes.
3. Who is sponsoring this clinical trial? If missing, state: "The document does not explicitly confirm the study sponsor."
4. What is the name of the drug being tested? What is its mechanism of action?
5. What is the total duration of the clinical trial for an individual participant (in days)?
6. What potential side effects and risks are listed in the document for participants?
7. What specific treatment or intervention is being tested in the trial?
8. What medical care is provided to participants during the trial? Mention routine procedures such as check-ups, lab tests, and monitoring.
9. What adverse events are mentioned in the document, and how are they monitored or managed?
10. What ethical guidelines does the trial follow? Include references to informed consent, IRB approval, and other relevant regulations.
11. What statistical methods are used for data analysis? If missing, state: "The document does not explicitly mention statistical methods."
12. Does the document provide any final conclusions or recommendations based on the study? If not, mention that results are pending.
13. What regulatory approvals or compliance measures are specified in the document?
14. What are the phases of the trial, and how many participants are involved in each phase? If missing, state: "The document does not explicitly mention trial phases."
15. Are there any insights into the participant experience mentioned in the document? Consider expectations, feedback, or participant rights.

"""
PROMPT_TEMPLATE = """
    You are an expert in clinical trials and medical research. 
    - Your task is to analyze the provided document and answer user queries in a formal, concise, and informative manner.
    - Use the provided document context to answer the questions.
    - Limit responses to two to three sentences per question.
    - If a specific detail is missing, state: "The document does not explicitly mention this information.
    - If the document does not explicitly mention a detail, state that the information is not available.
    - Do not infer or fabricate information beyond what is provided in the document.
    - Maintain clarity, accuracy, and a patient-friendly tone when necessary.
    
    Query: {user_query}
    Context: {document_context} 
    Answer:

"""

PDF_STORAGE_PATH = './document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama3.2:1b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="llama3.2:1b", params={"temperature": 0, "seed": 42, "top_k": 1})


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def load_doc_documents(file_path):
    # document_loader = Docx2txtLoader(file_path)
    document_loader = DoclingLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})

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


raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
processed_chunks = chunk_documents(raw_docs)
index_documents(processed_chunks)
relevant_docs = find_related_documents(user_query)
res = generate_answer(user_query, relevant_docs)
# print(res)
# Parse the structured response
parsed_answers = parse_answers(res)

# Print answers per question
image_num = 1
video_dict = dict()
text_dict = dict()
for question, answer in parsed_answers.items():
    print(f"{question}\n{answer.strip()}\n")
    prompt = f"create a cartoon image for {question}"
    image_name = f"./image_gen/{image_num}.png"
    # generate_infographic(prompt, image_name)
    image_num += 1
    video_dict[question]=image_name
    text_dict[question] = answer

print(text_dict)
video_gen(list(text_dict.values()), list(video_dict.values()))
