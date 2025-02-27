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
def create_knowlegebase(pdf):
    global knowledge_base, storage
    file_name_ext = Path(pdf).name
    file_name = file_name_ext.split('.')[0]
    knowledge_base = PDFKnowledgeBase(
        # path="./Algorithms_for_Interviews.pdf",
        # path="./clinical_protocol.pdf",
        path=pdf,
        vector_db=PgVector2(
            db_url=db_url,
            collection=f"protocol_db_{file_name}",
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

class Criteria(BaseModel):
    criteria: Optional[str] #= Field(..., description="this must be inclusion and exclusion criteria from the document")

class Queries(BaseModel):
     query: list[Criteria]


class Protocol2querygenerator(Workflow):
    # Define an Agent that will search the web for a topic
    
    docreader: Agent = Agent(
        model=Groq(id="llama3-70b-8192", embedder=embedder),
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        instructions=["extract  exclusion and inclusion criteria from knowledgebase.",
                    "if no information is available, alert the user that information is not available in the knowledgebase.",
                    ],
        # response_model=Criteria,
        structured_outputs=True,
        content_type="str",
        markdown=True,
    )

    # Define an Agent that will write the blog post
    sqlwriter: Agent = Agent(
        model=Groq(id="llama3-70b-8192", embedder=embedder),
        instructions=[
            "You will be provided with a text.",
            "Convert that text to oracle database SQL queries.",
            "If there no sufficient information, alert the user that you are not able to generate the SQL query.",
            "If there is sufficient information and you are not able to create SQL queries, alert the user with sufficient information why you are not able to generate.",
            "Always provide sources, do not make up information or sources.",
        ],
        markdown=True,
        show_tool_calls=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        structured_outputs=True,
        # response_model=Queries,

    )

    def run(self, topic: str, use_cache: bool = True) -> Iterator[RunResponse]:
            """This is where the main logic of the workflow is implemented."""

            # Step 1: Search the web for articles on the topic
            doc_content: Optional[Criteria] = self._get_content(topic)
            # If no search_results are found for the topic, end the workflow
            if doc_content is None:
                yield RunResponse(
                    event=RunEvent.workflow_completed,
                    content=f"Sorry, couldnot find anything on exclusion and inclusion criteria:",
                )
                return

            # Step 2: Write a blog post
            yield from self._write_sql_queries(doc_content)

    def _get_content(self, topic):
        try:
            document_content: RunResponse  = self.docreader.run(topic)
            print(document_content.content)
            return document_content.content
        except Exception as e:
             
             print(f"there is an error with docreader agent: {e}")

    
    def _write_sql_queries(self, doc_content) -> Iterator[RunResponse]:
        try:
             yield from self.sqlwriter.run(json.dumps({'text':doc_content}, indent=4), stream=True)
        except Exception as e:
             print(f"there is an error from sql writer agent: {e}")


def run_workflow(topic):
    # Run the workflow if the script is executed directly
    from rich.prompt import Prompt

    # # Convert the topic to a URL-safe string for use in session_id
    # url_safe_topic = topic.lower().replace(" ", "-")

    # Initialize the blog post generator workflow
    # - Creates a unique session ID based on the topic
    # - Sets up SQLite storage for caching results
    generate_sql = Protocol2querygenerator(
        session_id=f"converting protocol to sql",
        storage=PgWorkflowStorage(
            table_name="generate_sql_for_clinical_protocols_workflows",
            db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
        ),
    )

    # Execute the workflow with caching enabled
    # Returns an iterator of RunResponse objects containing the generated content
    blog_post: Iterator[RunResponse] = generate_sql.run(topic, use_cache=True)
    pprint_run_response(blog_post, markdown=True)

    # output_buffer = io.StringIO()
    # with redirect_stdout(output_buffer):
    #     pprint_run_response(blog_post, markdown=True)
    # output_string = output_buffer.getvalue()
    # 
    # out = list()
    # for resp in blog_post:
    #    out.append(resp.content)
    return blog_post

# if __name__ == "__main__":
#     from rich.prompt import Prompt

#     # Get topic from user
#     topic = Prompt.ask(
#         "[bold]Enter a something[/bold]\n✨",
#         default="analyze the document for exclusion and inclusion criteria for the most recent document in database",
#     )

#     # # Convert the topic to a URL-safe string for use in session_id
#     # url_safe_topic = topic.lower().replace(" ", "-")

#     # Initialize the blog post generator workflow
#     # - Creates a unique session ID based on the topic
#     # - Sets up SQLite storage for caching results
#     generate_sql = Protocol2querygenerator(
#         session_id=f"converting protocol to sql",
#         storage=PgWorkflowStorage(
#             table_name="generate_sql_for_clinical_protocols_workflows",
#             db_url="postgresql+psycopg://ai:ai@localhost:5532/ai",
#         ),
#     )

#     # Execute the workflow with caching enabled
#     # Returns an iterator of RunResponse objects containing the generated content
#     blog_post: Iterator[RunResponse] = generate_sql.run(topic, use_cache=True)

#     pprint_run_response(blog_post, markdown=True)