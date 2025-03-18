from langgraph.graph import StateGraph, START,END, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
# from IPython.display import Image, display
import streamlit as st
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.tools import tool, Tool


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="qwen-2.5-32b")

class State(TypedDict):
    """
    Represents the structure of the state used in the graph.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_decision: str


def pre_healthcheck_gudielines() -> dict:
    """
    Returns the guidelines for the patient. He/she should check before confirming an appointment.
    
    Args: None
    """
    guideline = f"""
                    - Your appointment will take about 2 hours and will take palce at the clinic you selected
                    in your eligibility screener.
                    
                    - During the appointment the study doctor will complete some health assessments to determine
                    if this study is right for you

                    - The assessments will include:
                        - Questions about your medical history
                        - A check of your vital signs (blodd pressure, heart rate, etc)
                        - Measurement of your height and weight
                        - A test for Covid-19
                        - A urine pregenance test for females 

                """
    st.session_state.input_required = True
    print(f"i am setting state to {st.session_state.input_required}")
    return {'messages':guideline}
    

def get_available_dates(date: Optional[str] = None)->dict:
    """
    shows the list of available dates for the appointment.

    Args: 
        date (Optional[str]): A specific date provided by the user. Default is None.

    """
    dates = ['2-MAY-2025', '10-MAY-2025', '31-MAY-2025']
    st.session_state.calendar = dates
    return {'messages':dates}


def get_available_locations(location: Optional[str] = None)->dict:
    """
    shows list of locations available for the patient for the appointment.

    Args: 
        location Optional[str]: A specific location provided by the user. Default is None.
    """
    sites = ['Hyderabad', 'Bangalore']
    st.session_state.site_decision = sites
    return  {'messages':sites}


def appoint_confirmation():
    """
    shows appointment confirmation.

    Args: None
    """
    conf = f"""
            your appointment is confirmed at {st.session_state.site_decision}, {st.session_state.date_decision}]
            """
    return {'messages':conf}

# Register tools
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_available_locations",
            "description": "list available locations for the clinical trial.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "location name",
                    }
                },
                "optional": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_available_dates",
            "description": "list available dates for the clinical trial.",
            "parameters": {
                "type": "object",
                "properties": {
                    "date": {
                        "type": "string",
                        "description": "date",
                    }
                },
                "optional": ["date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pre_healthcheck_gudielines",
            "description": "Guidelines or criterial for the clinical trail.",
            "parameters": {},
        },
    },

]
# tools = [
#     Tool(
#         name="locations",
#         func=get_available_locations,
#         description="list available locations for the clinical trial."

#     ),
#     Tool(
#         name="calendar",
#         func=get_available_dates,
#         description="List of available dates for the appointment."
#     ),
#     Tool(
#         name="prehealthcheck_gudielines",
#         func=pre_healthcheck_gudielines,
#         description="Guidelines or criterial for the clinical trail."
#     )
# ]

tools=[pre_healthcheck_gudielines,get_available_dates,get_available_locations]
llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=True)


def create_chat_graph():
    def process(state: State) -> dict:
        """
        Processes the input state and generates a response with tool integration.
        """

        System_message = """You are a digital health assistant that help schedule the appointment for the patient.
        Sequence should be:
            - pre health check guidelines or criteria.
            - get me available dates.
            - get me available locations.
        Patient Input : 
        """
        user_input = [System_message]+state["messages"]
        llm_response = llm_with_tools.invoke(user_input)
        return {"messages": [llm_response]}
    
    def human_approval(state: State) -> Command[Literal["chatbot", END]]:
        print(f"invoking humnan approval and awaiting for the result")
        print(f"st.session_state.user_decision :{st.session_state.user_decision}")
        is_approved = interrupt(
            {
                "question": "Is this correct?",
                # Surface the output that should be
                # reviewed and approved by the human.
                "llm_output": st.session_state.user_decision,
            }
        )
        if is_approved:
            return Command(goto="chatbot")
        else:
            return Command(goto=END)

    workflow = StateGraph(State)
    workflow.add_node("chatbot",process)
    # workflow.add_node("human_approval", human_approval)
    workflow.add_node("tools",ToolNode(tools))

    # Define conditional and direct edges
    workflow.add_edge(START,"chatbot")

    workflow.add_conditional_edges(
        "chatbot",
        # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
        # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
        tools_condition,
    )
    workflow.add_edge("tools","chatbot")
    # workflow.add_edge("chatbot","human_approval") #added this

    # Create memory saver for persistence
    memory = MemorySaver()
    
    # Compile the graph with memory
    chain = workflow.compile(checkpointer=memory)
    # png_data = chain.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_data)

    return chain

# agent = create_chat_graph()
# Define button callback function
def set_decision(decision):
    print(f"User selected the decision as : {decision}")
    st.session_state.user_decision = decision
    st.rerun()

st.set_page_config(page_title="🤖 Health chatbot", layout="wide")
st.header("🤖 Health bot" )
st.session_state.timeframe = ''
st.session_state.IsFetchButtonClicked = False
st.session_state.IsSDLC = False

# Session ID for this conversation
if "session_id" not in st.session_state:
    import uuid
    st.session_state.session_id = str(uuid.uuid4())

# Initialize or get chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if "input_required" not in st.session_state:
    st.session_state.input_required=''

if "user_decision" not in st.session_state:
    st.session_state.user_decision=''

if "date_decision" not in st.session_state:
    st.session_state.date_decision=''

if "calendar" not in st.session_state:
    st.session_state.calendar=''

if "site_decision" not in st.session_state:
    st.session_state.site_decision=''

# Initialize the graph on first run or when config changes
if "chat_graph" not in st.session_state :
    with st.spinner("Initializing chatbot..."):
        st.session_state.chat_graph = create_chat_graph()
        st.session_state.messages = []  # Clear messages on reset
        st.success("Chatbot initialized!")
    

# Display chat history
for message in st.session_state.messages:
    if isinstance(message, dict):  # Handle dict format
        role = message.get("role", "")
        content = message.get("content", "")
    else:  # Handle direct string format
        role = "user" if message.startswith("User: ") else "assistant"
        content = message.replace("User: ", "").replace("Assistant: ", "")
    
    print(message)
    print(role,'===', content)
    with st.chat_message(role):
        st.write(content)



# Input for new message
user_input = st.chat_input("Type your message here...")


if user_input:
    print("Running again")
    # Display user message
    with st.chat_message("user"):
        st.write(user_input)
    
    # Add to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Get response from the chatbot
    with st.spinner("Thinking..."):
        # Call the graph with the user's message
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        user_message = [HumanMessage(content=user_input)]
        result = st.session_state.chat_graph.invoke({"messages": user_message}, config)
        
        # Extract response
        response = result["messages"][-1].content
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)
    
    # Add to history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # state=st.session_state.chat_graph.get_state(st.session_state.session_id)
    # print(state.next)

    print(f"st.session_state.input_required - {st.session_state.input_required}")

if st.session_state.input_required:
    if st.session_state.user_decision=='':
        st.write("Do you want to proceed for appointment booking?")
        print("Do you want to proceed for appointment booking?")
        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes", on_click=set_decision, args=("yes",))  # Pass "yes" to callback
    
        with col2:
            st.button("No", on_click=set_decision, args=("no",))  # Pass "no" to callback
        st.stop()

if "user_decision" in st.session_state and st.session_state.user_decision:          
    print(f"st.session_state.user_decision ::: {st.session_state.user_decision}")
    if st.session_state.user_decision == 'yes':
        print("invoking graph after user decision")
        config = {"configurable": {"thread_id": st.session_state.session_id}}
        st.session_state.chat_graph.invoke(Command(resume=True), config=config)
        st.write("📜 Available Dates:")
        # Display choices in radio buttons
        selected_date = st.radio("Select a Date:", st.session_state["calendar"])

        print(f"Date selected : {st.session_state.date_decision}")
        if st.button("Submit Date"):
            st.session_state["date_decision"] = selected_date
            st.rerun()

        if st.session_state.date_decision:
            st.write("📜 Available Sites:")
            # Display choices in radio buttons
            selected_site  = st.radio("Select an option:", st.session_state["site_decision"])
            
            if st.button("Submit Choice"):
                st.session_state["site_decision"] = selected_site 
                st.success(f"✅ Appointment confirmed at {selected_site } on {selected_date}!")
elif st.session_state.user_decision=="no":
    st.warning("Appointment scheduling canceled.")


    # st.session_state.chat_graph.graph.update_state({"next_node": "pretone"})
    # st.session_state.chat_graph.graph.invoke()