from transformers import AutoModelForCausalLM, AutoTokenizer
from docx import Document
import torch
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
HF_TOKEN=os.environ['HF_TOKEN']

# Load the Mistral model
model_name = "BioMistral/BioMistral-7B"  # Use the desired model
model = AutoModelForCausalLM.from_pretrained(model_name,  token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name,  token=HF_TOKEN)

# Set pad_token to eos_token if it's not defined
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  

# Function to load and extract text from a DOCX file
def load_docx(file_path):
    doc = Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to chunk text if it is too long for the model
def chunk_text(text, max_tokens=1024):
    tokenized = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    num_tokens = tokenized['input_ids'].size(1)
    
    if num_tokens <= max_tokens:
        return [text]
    
    # Split the text into smaller chunks
    chunks = []
    paragraphs = text.split('\n')
    current_chunk = ""
    
    for para in paragraphs:
        temp_chunk = current_chunk + " " + para
        tokenized_temp = tokenizer(temp_chunk, return_tensors="pt", truncation=True, padding=True)
        if tokenized_temp['input_ids'].size(1) > max_tokens:
            if current_chunk:  # If the current chunk is not empty, add it to chunks
                chunks.append(current_chunk)
            current_chunk = para  # Start a new chunk with the current paragraph
        else:
            current_chunk = temp_chunk
    
    if current_chunk:  # Add the last chunk
        chunks.append(current_chunk)
    
    return chunks

# Function to get answers using the model and context
def answer_question_with_context(question, context):
    input_text = f"Context: {context}\nQuestion: {question}\nAnswer:"
    
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    # Create attention mask manually (1 for real tokens, 0 for padding tokens)
    attention_mask = inputs['input_ids'] != tokenizer.pad_token_id

    outputs = model.generate(inputs['input_ids'], attention_mask=attention_mask, max_length=1000, num_return_sequences=1)
    
    # Decode and return the answer
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer.split("Answer:")[-1].strip()

# Load the uploaded document
doc_text = load_docx('./HRP-503 - SAMPLE Biomedical Protocol.docx')

# Split the document into chunks if necessary
chunks = chunk_text(doc_text)

# Define the user query
user_query = "What is the main goal of the study?"

# Iterate over the chunks and get answers
for chunk in chunks:
    # print("Context Chunk:", chunk)
    answer = answer_question_with_context(user_query, chunk)
    print("Answer:", answer)
