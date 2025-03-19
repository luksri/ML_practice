import os
import streamlit as st
from langchain_groq import ChatGroq

class GroqLLM:
    def __init__(self,user_controls_input):
        self.user_controls_input=user_controls_input

    def get_llm_model(self):
        try:
            groq_api_key=self.user_controls_input['GROQ_API_KEY']
            selected_groq_model=self.user_controls_input['selected_groq_model']
            if groq_api_key=='' and os.environ["GROQ_API_KEY"] =='':
                st.error("Please Enter the Groq API KEY")

            llm = ChatGroq(api_key =groq_api_key, model=selected_groq_model)

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        return llm
    
    def get_llm_model_for_req(self):
        print("I am setting up model for requirements")
        try:
            groq_api_key=self.user_controls_input['GROQ_API_KEY']
            if groq_api_key=='' and os.environ["GROQ_API_KEY"] =='':
                st.error("Please Enter the Groq API KEY")

            llm = ChatGroq(api_key =groq_api_key, model="llama3-70b-8192")

        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        return llm
    
    def get_llm_model_for_code(self):
        pass