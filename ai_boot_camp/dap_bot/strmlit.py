import streamlit as st
import time

# Set page title
st.title("Guided Chatbot")

# Initialize chat history and question index in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'll guide you through a series of questions. Let's begin!"}
    ]
    st.session_state.current_question = 0

# Predefined questions
questions = [
    "What's your name?",
    "What's your favorite programming language?",
    "Have you used Streamlit before?",
    "What kind of app are you trying to build?"
]

# Predefined responses based on question index
def get_next_question(index, user_answer):
    responses = [
        f"Nice to meet you, {user_answer}! ",
        f"{user_answer} is a great choice! ",
        "Great to know! ",
        "That sounds like an interesting project! "
    ]
    
    # Return response to current question + next question (if available)
    if index < len(questions) - 1:
        return responses[index] + "Next question: " + questions[index + 1]
    else:
        return responses[index] + "Thanks for chatting with me! The conversation is complete."

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

# Get user input
if user_input := st.chat_input("Your answer:"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Get the next question/response
    next_response = get_next_question(st.session_state.current_question, user_input)
    
    # Increment the question counter
    st.session_state.current_question += 1
    
    # Display assistant response with a slight delay to simulate thinking
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        # Simulate typing
        for i in range(len(next_response) + 1):
            response_placeholder.markdown(next_response[:i] + "▌")
            time.sleep(0.01)
        response_placeholder.markdown(next_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": next_response})

# Option to restart the conversation
if st.button("Restart Conversation"):
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello! I'll guide you through a series of questions. Let's begin!"}
    ]
    st.session_state.current_question = 0
    st.rerun()