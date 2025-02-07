import streamlit as st
import os
from dotenv import load_dotenv
from tma_agents import run_workflow, Protocol2querygenerator


def user_input(user_question, agent):
    response = run_workflow(topic=user_question, agent=agent)
    st.write(response.content)


def main():
    st.set_page_config("TMA")
    st.header("TMA Parser")

    if "tma_agent" not in st.session_state:
        st.session_state.tma_agent=None

    
    UPLOAD_FOLDER = "./tmp/protocol_files"
    os.makedirs(UPLOAD_FOLDER, exist_ok=True) 

    st.title("📂 Upload File & Query LLM")
    pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button")
    if st.button("Submit & Process"):
        with st.spinner("Processing..."):
            if pdf_docs is not None:
                # Create a temporary file
                file_path = os.path.join(UPLOAD_FOLDER, pdf_docs.name)
                with open(file_path, "wb") as f:
                    f.write(pdf_docs.getbuffer())
                st.session_state.tma_agent=Protocol2querygenerator(file_path)
            st.success("Done")
    
    if st.session_state.tma_agent:
        print("tma agent got created")
    else:
        print("agent not created")

    user_question = st.text_input("Ask a Question from the PDF Files")
    if user_question and st.session_state.tma_agent:
        user_input(user_question, st.session_state.tma_agent)


if __name__ == "__main__":
    main()