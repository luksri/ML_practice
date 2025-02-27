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
knowledge_base=None
storage = None


### custom embedder since default is openAi
embedder = SentenceTransformerEmbedder(dimensions=384)

##### creating the knowledgebase for RAG
# def create_knowlegebase(pdf):
#     global knowledge_base, storage
# file_name_ext = Path(pdf).name
# file_name = file_name_ext.split('.')[0]
knowledge_base = PDFKnowledgeBase(
    # path="./Algorithms_for_Interviews.pdf",
    path="./clinical_protocol.pdf",
    # path=pdf,
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
knowledge_base.load(recreate=True, upsert=True)
storage=PgAgentStorage(table_name="pdf_assistant",db_url=db_url)


#### creating the agent
docreader: Agent = Agent(
        name='document reader',
        model=Groq(id="llama3-70b-8192", embedder=embedder),
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        # read_chat_history=True,
        # instructions=["extract  exclusion and inclusion criteria from knowledgebase.",
        #             "if no information is available, alert the user that information is not available in the knowledgebase.",
        #             ],
        # response_model=Criteria,
        structured_outputs=True,
        role='Searches Knowledgebase.',
        # markdown=True,
    )

sqlwriter: Agent = Agent(
        name='sql writer',
        model=Groq(id="llama3-70b-8192", embedder=embedder),
        # instructions=[
        #     "You will be provided with a text.",
        #     "Convert that text to oracle database SQL queries.",
        #     "If there no sufficient information, alert the user that you are not able to generate the SQL query.",
        #     "If there is sufficient information and you are not able to create SQL queries, alert the user with sufficient information why you are not able to generate.",
        #     "Always provide sources, do not make up information or sources.",
        # ],
        markdown=True,
        # show_tool_calls=True,
        # # Enable the assistant to read the chat history
        # read_chat_history=True,
        structured_outputs=True,
        # response_model=Queries,
        role='Write sql queries.',

    )
hn_team = Agent(
    name="Hackernews Team",
    model=Groq(id="llama3-70b-8192", embedder=embedder),
    team=[docreader, sqlwriter],
    instructions=[
        "First, use the docreader to search Knowledgebase for what the user is asking about.",
        "Then, ask the sqlwriter to write SQL queries for the output of docreader.",
        "Important: you must provide everything in user readable format.",
        "If not enough information is available, alert the user same.",
    ],
    show_tool_calls=True,
    markdown=True,
)
# create_knowlegebase(r'./clinical_protocol.pdf')
hn_team.print_response("Extract exclusion and inclusion criteria in human readable format from the document that is added to knowledgebase.", stream=True)
