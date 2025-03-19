import streamlit as st
import json
from src.langgraphagenticai.ui.streamlitui.loadui import LoadStreamlitUI
from src.langgraphagenticai.LLMS.groqllm import GroqLLM
from src.langgraphagenticai.graph.graph_builder import GraphBuilder
from src.langgraphagenticai.ui.streamlitui.display_result import DisplayResultStreamlit, DisplayResultStreamlit_sdlc
from src.langgraphagenticai.vectorstore.sdlc_vd_store import chroma_db

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
    user_input = ui.load_streamlit_ui()

    if not user_input:
        st.error("Error: Failed to load user input from the UI.")
        return

    
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = chroma_db()

    if 'req_added' not in st.session_state:
        st.session_state.req_added = False
    
    if 'user_story' not in st.session_state:
        st.session_state.user_story = False

    if 'graph' not in st.session_state:
        st.session_state.graph=None

    if 'stage_data' not in st.session_state:
        st.session_state.stage_data = None
    
    # if 'thread' not in st.session_state:
    #     st.session_state.thread={"configurable":{"thread_id":"sdlc"}}
    
    # if 'memory' not in st.session_state:
    #     st.session_state.memory = MemorySaver()

    if st.session_state.workflow_step == 1:
        # Text input for user message
        if st.session_state.IsFetchButtonClicked:
            user_message = st.session_state.timeframe 
        else :
            user_message = st.text_area("Enter your requirements:", height=200)

        # Add a submit button
        if st.button("Submit"):
            if user_message:
                st.success("Requirements submitted successfully!")
                st.session_state.req_added = True
                st.rerun()
            else:
                st.error("Please enter requirements before submitting.")
        try:
            # Configure LLM
            obj_llm_config = GroqLLM(user_controls_input=user_input)
            model = obj_llm_config.get_llm_model_for_req()
            
            if not model:
                st.error("Error: LLM model could not be initialized.")
                return

            # Initialize and set up the graph based on use case
            # usecase = user_input.get('selected_usecase')
            # if not usecase:
            #     st.error("Error: No use case selected.")
            #     return
            

            ### Graph Builder
            graph_builder=GraphBuilder(stage='requirements',model=model, vstore=st.session_state.vector_store
                                       )
            try:
                st.session_state.graph = graph_builder.setup_graph('sdlc')
                png_data = st.session_state.graph.get_graph().draw_mermaid_png()
                with open("graph.png", "wb") as f:
                    f.write(png_data)
                if st.session_state.req_added:
                    st.session_state.stage_data = st.session_state.graph.invoke({'messages': user_message})
                    # DisplayResultStreamlit(usecase,graph,user_message).display_result_on_ui()
            except Exception as e:
                st.error(f"Error: Graph setup failed - {e}")
                return
        except Exception as e:
            raise ValueError(f"Error Occurred with Exception : {e}")
        
    elif st.session_state.workflow_step == 2:
        st.session_state.user_story = True
        
    DisplayResultStreamlit_sdlc(graph=st.session_state.graph).display_sdlc_output(st.session_state.stage_data,st.session_state.workflow_step, st.session_state.req_added, st.session_state.user_story)
            

        



    
