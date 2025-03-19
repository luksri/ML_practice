import streamlit as st
from langchain_core.messages import HumanMessage,AIMessage,ToolMessage
import json


class DisplayResultStreamlit:
    def __init__(self,usecase,graph,user_message):
        self.usecase= usecase
        self.graph = graph
        self.user_message = user_message

    def display_result_on_ui(self):
        usecase= self.usecase
        graph = self.graph
        user_message = self.user_message
        if usecase =="Basic Chatbot":
                for event in graph.stream({'messages':("user",user_message)}):
                    print(event.values())
                    for value in event.values():
                        print(value['messages'])
                        with st.chat_message("user"):
                            st.write(user_message)
                        with st.chat_message("assistant"):
                            st.write(value["messages"].content)

        elif usecase=="Chatbot with Tool":
             # Prepare state and invoke the graph
            initial_state = {"messages": [user_message]}
            res = graph.invoke(initial_state)
            for message in res['messages']:
                if type(message) == HumanMessage:
                    with st.chat_message("user"):
                        st.write(message.content)
                elif type(message)==ToolMessage:
                    with st.chat_message("ai"):
                        st.write("Tool Call Start")
                        st.write(message.content)
                        st.write("Tool Call End")
                elif type(message)==AIMessage and message.content:
                    with st.chat_message("assistant"):
                        st.write(message.content)

        elif usecase =="sdlc":
                print("I am printing on the screen")
                try:
                    initial_state = {"messages": [user_message]}
                    res = graph.invoke(initial_state)
                    for message in res['messages']:
                        if type(message) == HumanMessage:
                            with st.chat_message("user"):
                                st.write(message.content)
                        elif type(message)==ToolMessage:
                            with st.chat_message("ai"):
                                st.write("Tool Call Start")
                                st.write(message.content)
                                st.write("Tool Call End")
                        elif type(message)==AIMessage and message.content:
                            with st.chat_message("assistant"):
                                st.write(message.content)
                        else:
                            st.write(message.content)
                    # for event in graph.stream({'messages':("user",user_message)}):
                    #     print(event.values())
                    #     for value in event.values():
                    #         print(value['messages'])
                    #         with st.chat_message("user"):
                    #             st.write(user_message)
                    #         with st.chat_message("assistant"):
                    #             st.write(value["messages"].content)
                except Exception as e:
                    print(f"there is an error in printing on UI: {e}")
                    for event in graph.stream(None, stream_mode="values"):
                        print(event)


class DisplayResultStreamlit_sdlc:
    def __init__(self, graph):
        self.graph = graph

    def display_sdlc_output(self, data, workflow_id, req_flag=False, story_flag=False):
        # print("\n\n data is \n\n")
        # print(data)
        if workflow_id == 1 and req_flag:
            st.write(data['user_requirements'])
        elif workflow_id == 2 and story_flag:
            st.write(data['user_stories'].content)
        # for event in self.graph.stream(None, {"configurable":{"thread_id":"sdlc"}}):
        #     print(event)