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
    def __init__(self, pdf):
        self.knowledge_base = PDFKnowledgeBase(
            # path="./Algorithms_for_Interviews.pdf",
            # path="./clinical_protocol.pdf",
            path=pdf,
            vector_db=PgVector2(
                db_url=db_url,
                collection=f"protocol_db",
                embedder=embedder,
            ),
            # chunking_strategy=AgenticChunking(),
            chunking_strategy=RecursiveChunking(),
            reader=PDFReader(chunk=True),
        )
        #### store the knowledgebase to database
        self.knowledge_base.load(recreate=True, upsert=True)
        self.storage=PgAgentStorage(table_name="pdf_assistant",db_url=db_url)
        self.docreader: Agent = Agent(
            model=Groq(id="llama3-70b-8192", embedder=embedder),
            knowledge_base=self.knowledge_base,
            storage=self.storage,
            # Show tool calls in the response
            show_tool_calls=True,
            # Enable the assistant to search the knowledge base
            search_knowledge=True,
            # Enable the assistant to read the chat history
            read_chat_history=True,
            instructions=[
                        "You are a text summarizer. Also a SQL query writer",
                        "extract  exclusion and inclusion criteria from knowledgebase.",
                        "if no information is available, alert the user that information is not available in the knowledgebase.",
                        "if you are not able to write sql queries, provide the reason."
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
