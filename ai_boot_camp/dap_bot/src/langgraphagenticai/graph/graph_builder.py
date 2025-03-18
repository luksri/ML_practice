from langgraph.graph import StateGraph, START,END, MessagesState
from langgraph.prebuilt import tools_condition,ToolNode
from langchain_core.prompts import ChatPromptTemplate
from src.langgraphagenticai.state.state import State
from src.langgraphagenticai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraphagenticai.nodes.chatbot_with_Tool_node import ChatbotWithToolNode
from src.langgraphagenticai.nodes.Health_chatbot_with_Tool_node import HealthChatbotWithToolNode
from src.langgraphagenticai.tools.serach_tool import get_tools,create_tool_node
from src.langgraphagenticai.tools.serach_tool_health import get_health_tools,create_health_tool_node




class GraphBuilder:

    def __init__(self,model, memory, session_data):
        self.llm=model
        self.graph_builder=StateGraph(State)
        self.memory = memory
        self.session_data = session_data
        self.user_decision = session_data['user_decision']  ## for yes/no for clinical trial
        self.date_decision = session_data['date_decision'] ## for date selection
        self.site_decision = session_data['site_decision'] ## for site choice
        # self.thread = {"configurable": {"thread_id": "Lakshman"}}

    def basic_chatbot_build_graph(self):
        """
        Builds a basic chatbot graph using LangGraph.
        This method initializes a chatbot node using the `BasicChatbotNode` class 
        and integrates it into the graph. The chatbot node is set as both the 
        entry and exit point of the graph.
        """
        self.basic_chatbot_node=BasicChatbotNode(self.llm)
        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot",END)


    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool integration.
        This method creates a chatbot graph that includes both a chatbot node 
        and a tool node. It defines tools, initializes the chatbot with tool 
        capabilities, and sets up conditional and direct edges between nodes. 
        The chatbot node is set as the entry point.
        """
        ## Define the tool and tool node

        tools=get_tools()
        tool_node=create_tool_node(tools)

        ##Define LLM
        llm = self.llm

        # Define chatbot node
        obj_chatbot_with_node = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_node.create_chatbot(tools)

        # Add nodes
        self.graph_builder.add_node("chatbot", chatbot_node)
        self.graph_builder.add_node("tools", tool_node)

        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_conditional_edges("chatbot", tools_condition)
        self.graph_builder.add_edge("tools","chatbot")

    def healthchatbot_with_tools_build_graph(self):
        """
        Build a basic health chatbot.
        """
        
        # Define chatbot node

        self.basic_chatbot_node=HealthChatbotWithToolNode(self.llm, self.session_data)
        
        # Add nodes
        self.graph_builder.add_node("chatbot",self.basic_chatbot_node.process)
        self.graph_builder.add_node("pretone", self.basic_chatbot_node.prehealthcheck_gudielines)
        self.graph_builder.add_node("calendar", self.basic_chatbot_node.calendar)
        self.graph_builder.add_node("site_locations", self.basic_chatbot_node.site_locations)
        self.graph_builder.add_node("appoint_confirmation", self.basic_chatbot_node.appoint_confirmation)


        # Define conditional and direct edges
        self.graph_builder.add_edge(START,"chatbot")
        self.graph_builder.add_edge("chatbot","pretone")
        self.graph_builder.add_conditional_edges(
                                                    "pretone",
                                                    lambda st: self.user_decision,
                                                    {
                                                        "yes": "calendar",
                                                        "no": END
                                                    }
                                                )
        
        self.graph_builder.add_edge("calendar","site_locations")
        self.graph_builder.add_edge("site_locations","appoint_confirmation")
        self.graph_builder.add_edge("appoint_confirmation",END)
    
    
    
    def setup_graph(self, usecase: str):
        """
        Sets up the graph for the selected use case.
        """
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()

        if usecase == "Chatbot with Tool":
            self.chatbot_with_tools_build_graph()
        
        if usecase == "HealthChatbot":
            self.healthchatbot_with_tools_build_graph()

        return self.graph_builder.compile()
    




    

