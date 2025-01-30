import streamlit as st
from phi.agent import Agent
from phi.model.google import Gemini
from phi.tools.duckduckgo import DuckDuckGo
from google.generativeai import upload_file, get_file
import google.generativeai as genai
import time
from pathlib import Path
import tempfile
from dotenv import load_dotenv

load_dotenv()
import os

API_KEY = os.getenv("GOOGLE_API_KEY")

if API_KEY:
    genai.configure(api_key=API_KEY)

# Page Configuration
st.set_page_config(
    page_title="Multimodal AI Agent Video Summarizer", page_icon="📽️", layout="wide"
)
st.title("Phidata Video AI Summarizer Agent 📽️")
st.header("Powered by Gemini 2.0 Flash EXp")

@st.cache_resource
def initialize_agent():
    return Agent(
        name="Video AI Summarizer",
        model=Gemini(id="gemini-2.0-flash-exp"),
        tools=[DuckDuckGo()],
        markdown=True,
    )

## Initial the agent
multimodal_Agent = initialize_agent()

# File Uploader
# video_file = st.video(
#     data="https://www.youtube.com/watch?v=iBnWHZmlIyY&list=PL_z_8CaSLPWekqhdCPmFohncHwz8TY2Go&index=6",
#     start_time=0
# )

link_in_prompt = st.text_input("Enter the video link that has to be processed")
if link_in_prompt:
        st.write("You entered: ", link_in_prompt)

if link_in_prompt:
    processed_video = st.video(link_in_prompt, format="video/mp4", start_time=0)
    user_query = st.text_area(
        "What insights are you seeking from this video?",
        placeholder="Ask anything about the video content. The AI agent will analyze and gather additional ",
        help="Provide specific questions or insights you want from the video.",
    )

    if st.button("🎤 Analyze video ", key="analyze video button"):
        if not user_query:
            st.warning("Please enter a question or insight to analyze the video.")
        else:
            try:
                # Upload and process video file
                # processed_video = video_file
                analysis_prompt = f"""
                    Analyze the uploaded video for content and context.
                    Respond to the following query using insights and supplementary web research
                    {user_query}
                    Provide a detailed , user-friendly, and actionable response.
                    """
                # AI Agent processing
                response = multimodal_Agent.run(
                    analysis_prompt, videos=[processed_video]
                )

                # Display the result
                st.subheader("Analysis Result")
                st.markdown(response.content)
            except Exception as error:
                st.error(f"An error occured during the analysis: {error}")
            finally:
                # Clean up temporary video file
                # Path(video_path).unlink(missing_ok=True)
                pass
else:
    st.info("Upload a video file to begin analysis.")

# Customize text area height
st.markdown(
    """
    <style>
    .stTextArea textarea {
        height: 100px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
