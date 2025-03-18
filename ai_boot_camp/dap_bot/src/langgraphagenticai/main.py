import streamlit as st
import json
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit

# MAIN Function START
def load_langgraph_agenticai_app():
    """
    Loads and runs the LangGraph AgenticAI application with Streamlit UI.
    This function initializes the UI, handles user input, configures the LLM model,
    sets up the graph based on the selected use case, and displays the output while 
    implementing exception handling for robustness.
    """
   
    # Load UI
    ui = LoadStreamlitUI()
    user_input, memory = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    # Text input for user message
    if st.session_state.IsFetchButtonClicked:
        user_message = st.session_state.timeframe 
    else :
        user_message = st.chat_input("Enter your message:")
    
    if "input_required" not in st.session_state:
        st.session_state.input_required=''
    
    if "user_decision" not in st.session_state:
        st.session_state.user_decision=''

    if "date_decision" not in st.session_state:
        st.session_state.date_decision=''
    
    if "site_decision" not in st.session_state:
        st.session_state.site_decision=''

    if user_message:
            try:
                # Configure LLM
                obj_llm_config = GroqLLM(user_controls_input=user_input)
                model = obj_llm_config.get_llm_model()
                
                if not model:
                    st.error("Error: LLM model could not be initialized.")
                    return

                # Initialize and set up the graph based on use case
                usecase = user_input.get('selected_usecase')
                if not usecase:
                    st.error("Error: No use case selected.")
                    return
                

                ### Graph Builder
                graph_builder=GraphBuilder(model, memory, st.session_state)
                try:
                    graph = graph_builder.setup_graph(usecase)
                    DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()
                except Exception as e:
                    st.error(f"Error: Graph setup failed - {e}")
                    return
                
            except Exception as e:
                 raise ValueError(f"Error Occurred with Exception : {e}")
    if st.session_state.input_required == True:
        st.write("Do you want to proceed for appointment booking?")
        choice = st.radio(
                        "Please select your choice",
                        ["Yes", "No"],
                        index=None,
                        key="yes_no_choice"
                    )

        if choice == "Yes":
            st.session_state.user_decision = "yes"
        elif choice == "No":
            st.session_state.user_decision = "no"
                                
    if st.session_state.date_decision:
        st.write("📜 Available Dates:")

        # Display choices in radio buttons
        selected_option = st.radio("Select an option:", st.session_state["date_decision"])
        
        if st.button("Submit Choice"):
            st.session_state["date_decision"] = selected_option

    if st.session_state.site_decision:
        st.write("📜 Available Sites:")

        # Display choices in radio buttons
        selected_option = st.radio("Select an option:", st.session_state["site_decision"])
        
        if st.button("Submit Choice"):
            st.session_state["site_decision"] = selected_option

   

    
