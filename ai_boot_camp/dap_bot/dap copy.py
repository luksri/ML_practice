from langgraph.graph import StateGraph, START,END, MessagesState
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
from typing import Annotated, Literal, Optional
from langgraph.graph.message import add_messages
from langchain_ollama import OllamaLLM
from IPython.display import Image, display
import streamlit as st
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver


load_dotenv()

llm = OllamaLLM(model="llama3.2:1b")

class State(TypedDict):
    """
    Represents the structure of the state used in the graph.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_decision: str

def create_chat_graph():
    def process(state: State) -> dict:
        """
        Processes the input state and generates a response with tool integration.
        """

        System_message = "Hello, I'm a digital assistant that can help schedule the first study appointment for you."
        user_input = [System_message]+state["messages"]
        llm_response = llm.invoke(user_input)
        return {"messages": [llm_response]}

    def prehealthcheck_gudielines(state: State) -> dict:
        """
        Returns the guidelines for the patient. He/she should check before confirming an appointment.
        """
        guideline = f"""
                        - Your appointment will take about 2 hours and wull take palce at the clinic you selected
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
        
    def calendar(state: State) ->dict:
        """
        Returns the list of available dates for the appointment
        """
        dates = ['2-MAY-2025', '10-MAY-2025', '31-MAY-2025']
        st.session_state.date_decision = dates
        return {'messages':dates}

    def site_locations(state: State) -> dict:
        """
        Returns list of locations available at for the patient
        """
        sites = ['Hyderabad', 'Bangalore']
        st.session_state.site_decision = sites
        return  {'messages':sites}

    def appoint_confirmation(state: State):
        """
        shows appointment confirmation
        """
        conf = f"""
                your appointment is confirmed at {st.session_state.site_decision}, {st.session_state.date_decision}]
                """
        return {'messages':conf}
    
    # Define conditional function for routing
    def route_based_on_status(state: State) -> str:
        """
        Determines the next node based on Streamlit session state.
        """
        
        user_status = st.session_state.get("user_decision", "None")  # Default to 'pending'
        print(f"i am here checking {user_status}")
        if user_status == "yes":
            return "yes"
        else:
            print("ending the graph")
            return "no"
        

    workflow = StateGraph(State)
    workflow.add_node("chatbot",process)
    workflow.add_node("pretone", prehealthcheck_gudielines)
    workflow.add_node("calendar", calendar)
    workflow.add_node("site_locations", site_locations)
    workflow.add_node("appoint_confirmation", appoint_confirmation)


    # Define conditional and direct edges
    workflow.add_edge(START,"chatbot")
    workflow.add_edge("chatbot","pretone")
    workflow.add_conditional_edges(
                                    "pretone",
                                    route_based_on_status,
                                    {
                                        "yes": "calendar",
                                        "no": END
                                    }
                                )

    workflow.add_edge("calendar","site_locations")
    workflow.add_edge("site_locations","appoint_confirmation")
    workflow.add_edge("appoint_confirmation",END)

    # Create memory saver for persistence
    memory = MemorySaver()
    
    # Compile the graph with memory
    chain = workflow.compile(interrupt_after=["pretone"],checkpointer=memory)
    png_data = chain.get_graph().draw_mermaid_png()

    with open("graph.png", "wb") as f:
        f.write(png_data)

    return chain


# Define button callback function
def set_decision(decision):
    st.session_state.user_decision = decision

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
    
    with st.chat_message(role):
        st.write(content)



# Input for new message
user_input = st.chat_input("Type your message here...")

if user_input:
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

    if st.session_state.input_required == True:
        st.write("Do you want to proceed for appointment booking?")
        # choice = st.radio(
        #                 "Please select your choice",
        #                 ["Yes", "No"],
        #                 index=None,
        #                 key="yes_no_choice"
        #             )

        # if choice == "Yes":
        #     st.session_state.user_decision = "yes"
        # elif choice == "No":
        #     st.session_state.user_decision = "no"
        # Buttons for decision

        col1, col2 = st.columns(2)
        with col1:
            st.button("Yes", on_click=set_decision, args=("yes",))  # Pass "yes" to callback
    
        with col2:
            st.button("No", on_click=set_decision, args=("no",))  # Pass "no" to callback


    # st.session_state.chat_graph.graph.update_state({"next_node": "pretone"})
    # st.session_state.chat_graph.graph.invoke()
    print(f"st.session_state.user_decision - {st.session_state.user_decision}")
    if st.session_state.user_decision == 'yes':
        st.write("📜 Available Dates:")

        # Display choices in radio buttons
        selected_option = st.radio("Select an option:", st.session_state["date_decision"])
        
        if st.button("Submit Choice"):
            st.session_state["date_decision"] = selected_option

    if st.session_state.date_decision:
        st.write("📜 Available Sites:")

        # Display choices in radio buttons
        selected_option = st.radio("Select an option:", st.session_state["site_decision"])
        
        if st.button("Submit Choice"):
            st.session_state["site_decision"] = selected_option




