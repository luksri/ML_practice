# """
# dont use any vector db. upload the doc and pass it as context
# """
import streamlit as st
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_docling import DoclingLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pandas as pd
from text_to_image_v2 import generate_infographic
import os
from docx import Document
# Use a pipeline as a high-level helper
from transformers import pipeline
import torch
# # Load model directly
# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForQuestionAnswering, AutoModelForCausalLM
# from transformers import BioGptTokenizer, BioGptForCausalLM, set_seed


# user_query =[
# "what is that one important reason one can participate this clinical trial?",
# "what is the goal of the study?",
# "what is the drug under study?",
# "explain how drug works?",
# "explain side effects and impacts w.r.to patient health?",
# "what are the risks involved?",
# "who is the sponsor for the study?",
# "what is the medicalcare provided?",
# "what is the trial period? (should be number of calendar days)",
# "(optional) do you have any statistics for the number of participants participating?",
# ]


# pipe = pipeline("text-generation", model="microsoft/biogpt")
# # tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
# # model = AutoModelForQuestionAnswering.from_pretrained("microsoft/biogpt")
# # model = AutoModelForCausalLM.from_pretrained("microsoft/biogpt")

# tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
# model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")



# # def load_doc_documents(file_path):
# #     # Load the document, extract text, tables, and save images
# #     text, tables = extract_text_and_images_and_tables(file_path)
# #     return text, tables

def load_doc_documents(file_path):
    document_loader = DoclingLoader(file_path)
    docs = document_loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    # tokenized_chunks = []
    # for chunk in chunks:
    #     if isinstance(chunk, str):
    #         # Tokenize the chunk (ensure it's a valid string)
    #         tokenized_chunks.append(tokenizer(chunk, padding=True, truncation=True, return_tensors="pt"))
    #     else:
    #         # raise ValueError("Each chunk must be of type string.")
    #         pass

    return chunks

# # Function to answer questions using BioGPT
# def answer_question(question, context):
#     # print(question, context) 
#     inputs = tokenizer(question, context, return_tensors="pt")
#     answer_start_scores, answer_end_scores = model(**inputs)

#     # Get the most likely beginning and end of the answer
#     start = torch.argmax(answer_start_scores)
#     end = torch.argmax(answer_end_scores)

#     # Decode the answer from the context
#     answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start:end+1]))

#     # set_seed(42)

#     # with torch.no_grad():
#     #     beam_output = model.generate(**inputs,
#     #                                 min_length=100,
#     #                                 max_length=1024,
#     #                                 num_beams=5,
#     #                                 early_stopping=True
#     #                                 )
#     # return tokenizer.decode(beam_output[0], skip_special_tokens=True)
#     return answer

# def summarize_with_llm(user_query, document_context):
#     """Retrieves relevant text using FAISS and summarizes it with an LLM."""
#     prompt = f"""
#     You are creating a short summary for a clinical trial. 

#     **Question:** {user_query}  
#     **Context:** {document_context}  
#     **Infographic Summary:** 
#     """
# #     prompt = f"""
# #     you are recruting participants for a clinical trial. 
# #     - you should serve to highlight important points that may be deciding factor for potential participants who may or may not want to join.
# #     - Finer details about doising, schedules etc should be limited.
# #     - Always refer to the individual as a 'participant.'
# #     - Use the provided context to answer the query in a Formal way. 
# #     - Answer should be user readble and should be limited to maximum of 2 sentences.
# #     - If unsure, state that you don't know.
# #     - Optional details can be omitted if there isn't enough information.


# # Query: {user_query}
# # Context: {document_context} 
# # Tables: {tables}
# # Answer:
# #     """
#     # print(retrieved_text)
#     # inputs = tokenizer(prompt, return_tensors="pt")#, truncation=True, max_length=512)
#     # output = LANGUAGE_MODEL.invoke(**inputs, max_new_tokens=200)
#     # response = tokenizer.decode(output[0], skip_special_tokens=True)

#     # print(response)
#     inputs = tokenizer(user_query, document_context, return_tensors="pt")
#     with torch.no_grad():
#         outputs = LANGUAGE_MODEL(**inputs)

#     answer_start = torch.argmax(outputs.start_logits)
#     answer_end = torch.argmax(outputs.end_logits) + 1
#     answer = tokenizer.convert_tokens_to_string(
#         tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
#     )
#     return answer

raw_docs = load_doc_documents('./HRP-503 - SAMPLE Biomedical Protocol.docx')
# generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
# # ai_response=summarize_with_llm(user_query, raw_docs)
# # print(ai_response)
# # response_dict = dict()
# # for each_q in user_query:
# #     x = summarize_with_llm(each_q, raw_docs)
# #     print(x)
# # print(response_dict)

# # df = pd.DataFrame(list(response_dict.items()), columns=['Query', 'Answer'])
# # # pd.ExcelWriter(df)
# # df.to_excel('qa_2.xlsx', index=False)
# # ai_response = summarize_with_llm(raw_docs)
# # print(ai_response)

# for question in user_query:
#     print(f"Question: {question}")
#     # Loop through chunks for context (could be more advanced with LangChain)
#     answers = []
#     for chunk in raw_docs:
#         answer = answer_question(question, chunk)
#         if answer.strip():  # If answer exists in the chunk, break early
#             answers.append(answer)
#     print(f"Answer: {' '.join(answers)}")
#     print(answers)



from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import os
load_dotenv()

os.environ['HF_TOKEN']=os.getenv("HF_TOKEN")
HF_TOKEN=os.environ['HF_TOKEN']

# Load BioMistral model and tokenizer
model_name = "BioMistral/BioMistral-7B"  # Use BioMistral or BioMistral-7B model
model = AutoModelForCausalLM.from_pretrained(model_name, token=HF_TOKEN)
tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)

# Provide context and a question
context = raw_docs
question = "What is the drug being tested in the clinical trial?"

# Combine context and question into one prompt
input_text = context + " Question: " + question

# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt")

# Generate an answer using the model
outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, do_sample=True)

# Decode and print the response
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
