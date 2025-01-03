import typer
from typing import Optional,List
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

# knowledge_base = PDFKnowledgeBase(
#     # path="./Algorithms_for_Interviews.pdf",
#     path="./ThaiRecipes.pdf",
#     vector_db=PgVector(
#         db_url=db_url,
#         table_name="ollama_embeddings",
#         embedder=embedder,
#     ),
#     reader=PDFReader(chunk=True),
# )



knowledge_base = PDFUrlKnowledgeBase(
    urls=["https://phi-public.s3.amazonaws.com/recipes/ThaiRecipes.pdf"],
    vector_db=PgVector2(collection="recipes", db_url=db_url, embedder=embedder)
)

#### store the knowledgebase to database
knowledge_base.load(recreate=True, upsert=True)
storage=PgAgentStorage(table_name="pdf_assistant",db_url=db_url)


#### creating the agent

def pdf_assistant(new: bool = False, user: str = "user"):
    run_id: Optional[str] = None

    # if not new:
    #      existing_run_ids: List[str] = storage.get_all_run_ids(user)
    #      if len(existing_run_ids) > 0:
    #          run_id = existing_run_ids[0]

    assistant = Agent(
        model=Groq(id="llama-3.3-70b-versatile", embedder=embedder),
        run_id=run_id,
        user_id=user,
        knowledge_base=knowledge_base,
        storage=storage,
        # Show tool calls in the response
        show_tool_calls=True,
        # Enable the assistant to search the knowledge base
        search_knowledge=True,
        # Enable the assistant to read the chat history
        read_chat_history=True,
    )
    if run_id is None:
        run_id = assistant.run_id
        print(f"Started Run: {run_id}\n")
    else:
        print(f"Continuing Run: {run_id}\n")

    assistant.cli_app(markdown=True)

if __name__=="__main__":
    typer.run(pdf_assistant)

