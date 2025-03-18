from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import ToolNode

def get_health_tools():
    """
    Return the list of tools to be used in the chatbot
    """
    tools=[TavilySearchResults]
    return tools

def create_health_tool_node(tools):
    """
    creates and returns a tool node for the graph
    """
    return ToolNode(tools=tools)


