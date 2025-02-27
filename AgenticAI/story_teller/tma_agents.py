import typer
from typing import Optional,List, Iterator
from phi.model.groq import Groq
from phi.agent import Agent, AgentKnowledge
from phi.storage.agent.postgres import PgAgentStorage
from phi.vectordb.pgvector import PgVector2, PgVector
from phi.embedder.ollama import OllamaEmbedder
from phi.knowledge.pdf import PDFKnowledgeBase, PDFReader, PDFUrlKnowledgeBase
# from phi.embedder.mistral import MistralEmbedder
# from phi.embedder.azure_openai import AzureOpenAIEmbedder
# from phi.embedder.huggingface import HuggingfaceCustomEmbedder
from phi.embedder.sentence_transformer import SentenceTransformerEmbedder
# from phi.embedder.openai import OpenAIEmbedder
from pydantic import BaseModel, Field
from phi.workflow import Workflow, RunResponse, RunEvent
from phi.storage.workflow.postgres import PgWorkflowStorage
import json
import io
from contextlib import redirect_stdout
from phi.utils.pprint import pprint_run_response
from phi.document.chunking.agentic import AgenticChunking
from phi.document.chunking.recursive import RecursiveChunking
from pathlib import Path
from phi.knowledge.docx import DocxKnowledgeBase, DocxReader
from phi.model.ollama import Ollama

import os
from dotenv import load_dotenv

load_dotenv()


# Access API keys
api_key = os.environ["MISTRAL_API_KEY"]
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"

### custom embedder since default is openAi
embedder = SentenceTransformerEmbedder(dimensions=384)





#### creating the agent
class Protocol2querygenerator:
    # Define an Agent that will search the web for a topic
    def __init__(self, pdf=None):
        self.knowledge_base = DocxKnowledgeBase(
            path="/Volumes/lucky-dev/ML_practice/AgenticAI/story_teller/HRP-503 - SAMPLE Biomedical Protocol.docx",
             vector_db=PgVector(
                db_url=db_url,
                table_name=f"docx_documents",
                embedder=embedder,
            ),
            # chunking_strategy=AgenticChunking(),
            chunking_strategy=RecursiveChunking(),
            reader=DocxReader(chunk=True),
        )
        #### store the knowledgebase to database
        self.knowledge_base.load(recreate=True, upsert=True)
        self.storage=PgAgentStorage(table_name="doc_assistant",db_url=db_url)
        self.docreader: Agent = Agent(
            # model=Groq(id="mixtral-8x7b-32768", embedder=embedder),
            model=Ollama(id="mistral", embedder=embedder),
            knowledge_base=self.knowledge_base,
            storage=self.storage,
            # Show tool calls in the response
            show_tool_calls=True,
            # Enable the assistant to search the knowledge base
            search_knowledge=True,
            # Enable the assistant to read the chat history
            read_chat_history=True,
            instructions=[
                        """
1. Document Understanding
    Read the entire IRB document carefully.
    Identify key sections, including:
        Study Title,
        Principal Investigator (PI) & Contact Information,
        Purpose of the Study,
        Eligibility Criteria (Inclusion & Exclusion),
        Study Procedures,
        Risks & Benefits,
        Informed Consent Process,
        Confidentiality & Data Protection,
        Voluntary Participation & Withdrawal Rights,
        Compensation (if applicable),
        IRB Approval & Expiry Date,
        procedures involved,
        confidentiality measures,

2. Text Processing & Summarization
    Generate a structured summary highlighting the essential details.
    Convert complex medical or legal terms into simple, participant-friendly language while maintaining accuracy.
    Ensure that the summary retains all ethical considerations outlined in the document.

3. Compliance Check
    Verify that the informed consent section explicitly mentions:
        Right to withdraw at any time without penalty.
        Potential risks & side effects.
        How data will be stored & protected.
    Ensure that the study complies with ethical guidelines such as:
        Belmont Report Principles (Respect, Beneficence, Justice).
        HIPAA (if applicable) for data privacy.
        FDA/ICH GCP (for clinical trials).

4. Transformation for Participant Engagement
    Convert extracted information into:
        A participant-friendly FAQ.
        A persuasive video script for recruitment. 
        A simple infographic-style summary.

# 5. Generate a Video Script (Optional)
#     Based on the IRB document, create a short, engaging narrative to explain the study to potential participants.
#     Ensure the script covers:
#         What the study is about.
#         Why the participant is important.
#         What they can expect.
#         Their rights & safety assurances.
Note: use 'Participant' instead of patient.
"""
                        ],
            # response_model=Criteria,
            structured_outputs=True,
            content_type="str",
            markdown=True,
        )
        
def run_workflow(topic, agent):
    # Run the workflow if the script is executed directly

    # generate_sql = Protocol2querygenerator()
    response_stream: Iterator[RunResponse] = agent.docreader.run(topic)
    pprint_run_response(response_stream, markdown=True)


    return response_stream

agent = Protocol2querygenerator()
topic = """
You are an agent for IRB-approved clinical trials. 
use knowledge base for every question below.
- what is the objective of the document?
- why it is required to participate in the study?
- what is the drug under test?
- what are the impacts of the drug and how it is adminstered?

"""
run_workflow(topic=topic, agent=agent)
