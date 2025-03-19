from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from typing import TypedDict, Annotated, List
from langchain_core.messages import HumanMessage, AIMessage

class State(TypedDict):
    """
    Represents the structure of the state used in the graph.
    """
    messages: Annotated[list, add_messages]
    user_requirements: List[str]
    user_stories: str