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
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal
import time
from langgraph.types import interrupt, Command
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition
from langchain.tools import tool


load_dotenv()
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
llm=ChatGroq(model="qwen-2.5-32b")

class State(TypedDict):
    """
    Represents the structure of the state used in the graph.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    user_decision: str

def prehealthcheck_gudielines() -> dict:
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
    return {'messages':guideline}
    
def calendar() ->dict:
    """
    Returns the list of available dates for the appointment
    """
    dates = ['2-MAY-2025', '10-MAY-2025', '31-MAY-2025']
    # date = user_selection(dates)
    return {'messages':dates}

def locations() -> dict:
    """
    Returns list of locations available.
    """
    sites = ['Hyderabad', 'Bangalore']
    return  {'messages':sites}

def appoint_confirmation():
    """
    shows appointment confirmation
    """
    conf = f"""
            Ask user to choose date and site for appointment confirmation
            """
    return {'messages':conf}

def user_choice_selection(options):
    """
    Input method to choose a choice for the provided options
    """
    # Display options
    print("Choose an option:")
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")

    # Get user choice
    choice = input("Enter the number of your choice: ")

    # Validate input
    if choice.isdigit():
        choice = int(choice)
        if 1 <= choice <= len(options):
            print(f"You selected: {options[choice - 1]}")
        else:
            print("Invalid choice, please select a valid number.")
    else:
        print("Invalid input, please enter a number.")
    return choice


tools=[prehealthcheck_gudielines,calendar,locations]

llm_with_tools=llm.bind_tools(tools,parallel_tool_calls=False)


def process(state: State) -> dict:
    """
    Processes the input state and generates a response with tool integration.
    """

    System_message = "Hello, I'm a digital assistant that can help schedule the first study appointment for you."
    user_input = [System_message]+state["messages"]
    llm_response = llm_with_tools.invoke(user_input)
    return {"messages": [llm_response]}

def create_chat_graph():
    
    workflow = StateGraph(State)
    workflow.add_node("chatbot",process)
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

    # Compile the graph with memory
    chain = workflow.compile()
    png_data = chain.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(png_data)

    return chain

def tool_message_read(res):
    msg=None
    for message in res['messages']:
        if type(message) == HumanMessage:
                pass
        elif type(message)==ToolMessage:
                msg = eval(message.content)
        elif type(message)==AIMessage and message.content:
                pass
    return msg['messages']


# Set page title
st.title("Guided Chatbot")

# Initialize chat history and question index in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello, I am a scheduling assistant for clinical trail assisting you with booking an appointment."}
    ]
    st.session_state.current_question = 0
    st.session_state.llm_messages = []
    st.session_state.conversation_active = True
    st.session_state.appointment_details = []

# Predefined questions
questions = ["What's your name?",
            "show me the guidelines or prerequisites for the clinical trail.",
            "ready for trial",
            "show me the available dates.",
            "show me the available locations.",
            "appointment confirmation"
            ]
# Define which questions should be answered by LLM
llm_questions = [1,3,4]  # Questions at index 2 and 3 will use LLM


# Function to get next question or response
def get_next_question(index, user_answer):
    if index < len(questions)-1:
        return f"Next question: {questions[index + 1]}"
    else:
        
        return "Thanks for chatting with me! The conversation is complete."

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Only show the next question if we haven't finished all questions
if st.session_state.current_question < len(questions):
    # If this is a new question, add it to the chat
    if st.session_state.current_question >= 0 and len(st.session_state.messages) == (st.session_state.current_question * 2) + 1:
        # Add the question to chat history
        st.session_state.messages.append({"role": "assistant", "content": questions[st.session_state.current_question]})
        # Display the question
        with st.chat_message("assistant"):
            st.markdown(questions[st.session_state.current_question])

# Initialize the graph on first run or when config changes
if "chat_graph" not in st.session_state :
    with st.spinner("Initializing chatbot..."):
        st.session_state.chat_graph = create_chat_graph()
        st.success("Chatbot initialized!")

# Check if conversation is still active
if not st.session_state.conversation_active:
    st.info("The conversation has ended. Click 'Restart Conversation' to begin again.")


# Get user input
if user_input := st.chat_input("Your answer:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Check if the current question should be answered by LLM
    current_q = st.session_state.current_question
    
    # Display assistant response
    with st.chat_message("assistant"):
        if current_q in llm_questions:
            # Update OpenAI message history
            st.session_state.llm_messages.append({"role": "user", "content": questions[current_q]})
            
            res = st.session_state.chat_graph.invoke({"messages": questions[current_q]})
            response = tool_message_read(res)
            if type(response) == list:
                selected_option = st.radio("Select an option:", response, key='selection')
                # selected_option = st.selectbox("Select one", response,key="selection")
                if st.button('Submit'):
                    response = selected_option
                    print(f"selected response: {selected_option}")
                st.session_state.appointment_details.append(response)
            else:
                response=response
            # Save the response to OpenAI history
            st.session_state.llm_messages.append({"role": "assistant", "content": response})
            
            # Add the next question
            next_q = get_next_question(current_q, user_input)
            st.write(next_q)
            
            # Combine for history
            print(response)
            full_response = response + "\n\n" + next_q
            
        else:
            # Regular predefined response
            response = f"Thanks for sharing that you {user_input}! "
            next_q = get_next_question(current_q, user_input)
            print(questions[current_q], user_input, next_q, len(questions))
            if current_q == 2:
                if user_input.lower() == 'yes':
                    response = "Yes"
                else:
                    response = "No"
                    st.session_state.conversation_active = False
                    st.rerun()
            if current_q == len(questions) - 1:
                print("i am updating confirmation")
                print(st.session_state.appointment_details)
                message = "Your appointment is confirmed on "+ st.session_state.appointment_details[0]+" at "+ st.session_state.appointment_details[1]
                print(message)
                st.write(message)

            full_response = response + next_q

            # Display with typing effect
            response_placeholder = st.empty()
            for i in range(len(full_response) + 1):
                response_placeholder.markdown(full_response[:i] + "▌")
                time.sleep(0.01)
            response_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
    
    # Increment the question counter
    st.session_state.current_question += 1
    

# Option to restart the conversation
if st.button("Restart Conversation"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'll guide you through a series of questions. Some answers will be generated by an AI. Let's begin!"}
    ]
    st.session_state.current_question = 0
    st.session_state.openai_messages = []
    st.session_state.conversation_active = True
    st.rerun()


# st.set_page_config(page_title="🤖 Health chatbot", layout="wide")
# st.header("🤖 Health bot" )
# st.session_state.timeframe = ''
# st.session_state.IsFetchButtonClicked = False
# st.session_state.IsSDLC = False

# # Session ID for this conversation
# if "session_id" not in st.session_state:
#     import uuid
#     st.session_state.session_id = str(uuid.uuid4())

# # Initialize or get chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Initialize session state
# if "user_decision" not in st.session_state:
#     st.session_state.user_decision = None
# if "selected_choice" not in st.session_state:
#     st.session_state.selected_choice = None

# # Initialize the graph on first run or when config changes
# if "chat_graph" not in st.session_state :
#     with st.spinner("Initializing chatbot..."):
#         st.session_state.chat_graph = create_chat_graph()
#         st.session_state.messages = []  # Clear messages on reset
#         st.success("Chatbot initialized!")
    

# # Display chat history
# for message in st.session_state.messages:
#     if isinstance(message, dict):  # Handle dict format
#         role = message.get("role", "")
#         content = message.get("content", "")
#     else:  # Handle direct string format
#         role = "user" if message.startswith("User: ") else "assistant"
#         content = message.replace("User: ", "").replace("Assistant: ", "")
    
#     with st.chat_message(role):
#         st.write(content)



# # Input for new message
# user_input = st.chat_input("Type your message here...")

# if user_input:
#     # Display user message
#     with st.chat_message("user"):
#         st.write(user_input)
    
#     # Add to history
#     st.session_state.messages.append({"role": "user", "content": user_input})
    
#     # Get response from the chatbot
#     with st.spinner("Thinking..."):
#         # Call the graph with the user's message
#         for each_message in message_list:
#             # user_message = [HumanMessage(content=user_input)]
#             print(each_message)
#             if type(each_message)==AIMessage:
#                 response = each_message.content
#             elif each_message == "ready for trial":
#                 # Display Yes/No buttons
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     if st.button("Yes"):
#                         st.session_state.selected_choice = "Yes"
#                 with col2:
#                     if st.button("No"):
#                         st.session_state.selected_choice = "No"

#                 # Display the submit button **only if** a choice has been made
#                 if st.session_state.selected_choice:
#                     if st.button("Submit"):
#                         st.session_state.user_decision = st.session_state.selected_choice
#                         st.write(f"You selected: {st.session_state.user_decision}")
#                         st.session_state.selected_choice = None  # Reset for next interaction
#                         st.experimental_rerun()  # Rerun the script to refresh state
                
#             else:
#                 result = st.session_state.chat_graph.invoke({"messages": each_message})
#                 # Extract response
#                 # response = result["messages"][-1].content
#                 # print(result["messages"][-1].content)
#                 # print(result)
#                 tool_msg = tool_message_read(result)
#                 tool_msg = tool_msg['messages']
#                 # print(tool_msg)
#                 if type(tool_msg) == list:
#                     selected_option = st.radio("Select an option:", tool_msg)
#                     response = selected_option
#                 else:
#                     response=tool_msg
        
#             # Display assistant response
#             with st.chat_message("assistant"):
#                 st.write(response)
    
#     # Add to history
#     st.session_state.messages.append({"role": "assistant", "content": response})

   


