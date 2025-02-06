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
from phi.utils.pprint import pprint_run_response

import os
from dotenv import load_dotenv

load_dotenv()


# Access API keys
api_key = os.environ["MISTRAL_API_KEY"]
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
db_url = "postgresql+psycopg://ai:ai@localhost:5532/ai"


### custom embedder since default is openAi
# embedder = MistralEmbedder(api_key=api_key)
# embedder = OllamaEmbedder(dimensions=4096, model="mxbai-embed-large")
embedder = SentenceTransformerEmbedder(dimensions=384)

##### creating the knowledgebase for RAG

knowledge_base = PDFKnowledgeBase(
    # path="./Algorithms_for_Interviews.pdf",
    path="./clinical_protocol.pdf",
    vector_db=PgVector2(
        db_url=db_url,
        collection="protocol_db",
        embedder=embedder,
    ),
    reader=PDFReader(chunk=True),
)



# knowledge_base = PDFUrlKnowledgeBase(
#     urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
#     vector_db=PgVector2(collection="recipes", db_url=db_url, embedder=embedder)
# )

#### store the knowledgebase to database
knowledge_base.load(recreate=True, upsert=True)
storage=PgAgentStorage(table_name="pdf_assistant",db_url=db_url)


#### creating the agent

# def pdf_assistant(new: bool = False, user: str = "user"):
#     run_id: Optional[str] = None

#     assistant = Agent(
#         model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
#         run_id=run_id,
#         user_id=user,
#         knowledge_base=knowledge_base,
#         storage=storage,
#         # Show tool calls in the response
#         show_tool_calls=True,
#         # Enable the assistant to search the knowledge base
#         search_knowledge=True,
#         # Enable the assistant to read the chat history
#         read_chat_history=True,
#     )
#     if run_id is None:
#         run_id = assistant.run_id
#         print(f"Started Run: {run_id}\n")
#     else:
#         print(f"Continuing Run: {run_id}\n")

#     assistant.cli_app(markdown=True)

# if __name__=="__main__":
#     typer.run(pdf_assistant)

class Criteria(BaseModel):
    criteria: Optional[str] = Field(..., description="this must be inclusion and exclusion criteria from the document")


class Protocol2querygenerator(Workflow):
    # Define an Agent that will search the web for a topic
    docreader: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
        instructions=["Search the knowledgebase for the most recent document uploaded.",
                      "Read and analyse that document.",
                     "Extract exclusion and inclusion criteria in a user readable format."],
        # response_model=Criteria,
        structured_outputs=True,
        markdown=True,
    )

    # Define an Agent that will write the blog post
    sqlwriter: Agent = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
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
        # response_model=Criteria,

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
            document_content = self.docreader.run(topic)
            # print(document_content)
            # print([v.model_dump() for v in document_content.content.criteria])
            return document_content.content
        except Exception as e:
             print(f"there is an error with docreader agent: {e}")

    
    def _write_sql_queries(self, doc_content) -> Iterator[RunResponse]:
        try:
             yield from self.sqlwriter.run(json.dumps({'text':doc_content}, indent=4), stream=True)
        except Exception as e:
             print(f"there is an error from sql writer agent: {e}")


# Run the workflow if the script is executed directly
if __name__ == "__main__":
    from rich.prompt import Prompt

    # Get topic from user
    topic = Prompt.ask(
        "[bold]Enter a something[/bold]\n✨",
        default="analyze the document for exclusion and inclusion criteria for the most recent document in database",
    )

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

    # print(blog_post, type(blog_post))
    # Print the response
    pprint_run_response(blog_post, markdown=True)