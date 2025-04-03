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
from langchain_community.vectorstores import FAISS
from text_to_image_v2 import generate_infographic
from video_gen import video_gen
from pydantic import BaseModel, Field
from typing import Optional, List
from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
HF_TOKEN=os.environ['HF_TOKEN']

# You are an expert summarizer. Summarize the key insights from the retrived document context.
#     - summary should not exceed more than 3 sentences
#     - provide source information (Section numbers) along with summary
#     - if information is not available respond to user as **not enough information is available in the document provided**
                          
#     context: {document_text}
#     Question : {text}

# user_query = """ 
# what is the document all about?
# """

EMBED_MODEL_ID = "deepseek-ai/DeepSeek-R1"
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="llama3.1", params={"temperature": 0, "seed": 42, "top_k": 0, "top_p":0.9})


def load_doc_documents(file_path):
    document_loader = DoclingLoader(file_path,export_type=ExportType.MARKDOWN)
    docs = document_loader.load()
    # loader=PyPDFLoader(file_path)
    # docs=loader.load()
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1500,chunk_overlap=100)
    final_documents=text_splitter.split_documents(docs)
    return final_documents


def get_embedding(docs):
    """Generates an embedding for the given text."""
    vectordb=FAISS.from_documents(docs,EMBEDDING_MODEL)
    print(f"Size of vector DB : {vectordb.index.ntotal}")
    retriever=vectordb.as_retriever()
    return retriever

def summarize_with_llm(text):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""
    # global vstore

    retrieved_text=st.session_state.vstore.invoke(text)
    document_text=[i.page_content for i in retrieved_text]
    document_text="/n".join(document_text)
    
    prompt = f"""
    "Summarize the given user manual **Context** while preserving key details. Include section number references where relevant. 
    Structure the response with bullet points for clarity. 
    Ensure the summary is concise yet informative. 
    Format the output as follows:

        📌 Summary: [Concise explanation]
        🔢 Source Reference: Section [X.Y]

    Use markdown formatting for readability. If multiple sections contribute to a point, list all relevant section numbers."**
    
    User Interaction :
        Once the context is processed, you should respond accurately to user queries. If a query is outside the provided context, state so clearly.
    
    context: {document_text}
    question: {text}
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


# ✅ Ensure memory is stored in session_state
if "raw_docs" not in st.session_state:
    st.session_state.raw_docs = load_doc_documents('./hp_manual.pdf')


if "vstore" not in st.session_state:
    st.session_state.vstore = get_embedding(st.session_state.raw_docs)

# raw_docs = load_doc_documents('./user_manual.pdf')
# vstore = get_embedding(raw_docs)


# Streamlit Interface
st.title("User Manual Helper")
st.write("Ask me anything about the product.")

# User Input
user_input = st.text_input("Enter your question:")

# Display Response
if user_input:
    with st.spinner("Thinking..."):
        response = summarize_with_llm(user_input)
        # response.
        
        # parsed_answers = parse_answers(response)
        st.markdown(response)
