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

user_query =[
"what is that one important reason one can participate this clinical trial?",
"what is the goal of the study?",
"what is the drug under study?",
"explain how drug works?",
"explain side effects and impacts w.r.to patient health?",
"what are the risks involved?",
"who is the sponsor for the study?",
"what is the medicalcare provided?",
"what is the trial period? (should be number of calendar days)",
"(optional) do you have any statistics for the number of participants participating?",
]

PROMPT_TEMPLATE = """
you are recruting participants for a clinical trial. 
- your task is to provide summary limited to maximum of 2 sentences using the given Context {document_context} 
- you should serve to highlight important points that may be deciding factor for potential participants who may or may not want to join.
- Finer details about doising, schedules etc should be limited.
- Always refer to the individual as a 'participant.'
- If unsure, state that you don't know.
- Optional details can be omitted if there isn't enough information.



"""

PDF_STORAGE_PATH = './document_store/pdfs/'
EMBEDDING_MODEL = OllamaEmbeddings(model="llama2:13b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
# LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b", params={"temperature": 0, "seed": 42, "top_k": 1})
LANGUAGE_MODEL = OllamaLLM(model="llama2:13b", params={"temperature": 0, "seed": 42, "top_k": 1})


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

raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
processed_chunks = chunk_documents(raw_docs)
index_documents(processed_chunks)
response_dict = dict()
for each_q in user_query:
    relevant_docs = find_related_documents(each_q)
    response_dict[each_q] = generate_answer(each_q, relevant_docs)
# print(response_dict)

df = pd.DataFrame(list(response_dict.items()), columns=['Query', 'Answer'])
# pd.ExcelWriter(df)
df.to_excel('qa.xlsx', index=False)

# # UI Configuration


# st.title("📘 DocuMind AI")
# st.markdown("### Your Intelligent Document Assistant")
# st.markdown("---")

# # File Upload Section
# uploaded_pdf = st.file_uploader(
#     "Upload Research Document (PDF)",
#     type=["pdf", "docx"],
#     help="Select a PDF/docx document for analysis",
#     accept_multiple_files=False

# )

# if uploaded_pdf:
#     saved_path = save_uploaded_file(uploaded_pdf)
#     raw_docs = load_doc_documents(saved_path)
#     processed_chunks = chunk_documents(raw_docs)
#     index_documents(processed_chunks)
    
#     st.success("✅ Document processed successfully! Ask your questions below.")
    
#     user_input = st.chat_input("Enter your question about the document...")
    
#     if user_input:
#         with st.chat_message("user"):
#             st.write(user_input)
        
#         with st.spinner("Analyzing document..."):
#             relevant_docs = find_related_documents(user_input)
#             ai_response = generate_answer(user_input, relevant_docs)
            
#         with st.chat_message("assistant", avatar="🤖"):
#             st.write(ai_response)
