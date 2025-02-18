"""
This version uses vector db + hybrid search

NOT working
"""


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
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceHubEmbeddings
from transformers import AutoModel, AutoTokenizer

# Load BioBERT manually
model_name = "dmis-lab/biobert-base-cased-v1.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Load embeddings (BioBERT for better clinical text understanding)
embeddings = HuggingFaceEmbeddings(model_name="dmis-lab/biobert-base-cased-v1.2", model_kwargs={"device": "cpu"})
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


def load_doc_documents(file_path):
    # document_loader = Docx2txtLoader(file_path)
    document_loader = DoclingLoader(file_path)
    docs = document_loader.load()
    # Split into chunks (better for retrieval)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_embedding(docs):
    """Generates an embedding for the given text."""
    vectordb=FAISS.from_documents(docs,embeddings)
    retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.3} )
    # retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 5, "score_threshold": 0.75} )
    return retriever

def hybrid_search(query, retriever, keyword_filter=None):
    # Retrieve relevant documents from FAISS
    retrieved_docs = retriever.get_relevant_documents(query)
    
    # Combine retrieved docs
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    
    # Apply keyword filtering
    if keyword_filter:
        filtered_context = "\n\n".join([doc for doc in context.split("\n\n") if keyword_filter.lower() in doc.lower()])
        if filtered_context:
            return filtered_context  # Return only filtered content if found
    
    return context  # Return full FAISS-retrieved content if no keyword match


def summarize_with_llm(text, vstore):
    """Retrieves relevant text using FAISS and summarizes it with an LLM."""
    context = hybrid_search(text, vstore)
    
    # Avoid hallucination if no context is found
    if not context.strip():
        return "The document does not explicitly mention this information."

    # Structured prompt
    
    prompt = f"""
    Using the given clinical trial document excerpts, provide a structured response.
    
    **Query:** {text}
    **Context:**
    {context}
    
    **Answer:**
    """
    # print(retrieved_text)
    response = LANGUAGE_MODEL.invoke(prompt)
    
    return response
    

# Example Usage
docx_path = "./HRP-503 - SAMPLE Biomedical Protocol.docx"  # Replace with actual file path
text_data = load_doc_documents(docx_path)

# Initialize FAISS index (dimensionality must match embedding size, e.g., 384 for MiniLM)
vstore = get_embedding(text_data)
summary = summarize_with_llm(user_query, vstore)
# summary = summarize_with_llm(f"clinical trial for HRP-503 SAMPLE Biomedical Protocol", vstore)
print(summary)

# for each_q in user_query:
#     summary = summarize_with_llm(each_q, vstore)
#     print(summary)

