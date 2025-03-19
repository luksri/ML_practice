from src.langgraphagenticai.state.state import State
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage

class SDLCNode:
    """
    Basic chatbot logic implementation.
    """
    def __init__(self, vstore, requirements_model=None):
        # if requirements_model:
        #     self.llm = requirements_model
        # else:
        #     print("No LLM is set")
        #     raise Exception("!!!!!! Set up an LLM before you continue !!!!!")
        self.llm = requirements_model
        self.vstore = vstore

    @staticmethod
    def unpack_messages(res):
        """
        this method unpacks the human messages from the prompt
        """
        msg=None
        print(res)
        for message in res:
            if type(message) == HumanMessage:
                    msg = message.content
                    msg = eval(msg)
            elif type(message)==ToolMessage:
                    pass
            elif type(message)==AIMessage:
                    msg = message.content
        # print(msg)
        # if msg:
        #      msg = eval(msg)
        return msg
    
    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a chatbot response.
        """
        return {"messages":self.llm.invoke(state['messages'])}
    
    def user_requirements(self, state: State) -> dict:
        """
        Processes the input state for Requirements
        """
        print("i am here at user requirements")
        requirements = SDLCNode.unpack_messages(state['messages'])
        # print(state['messages'], type(state['messages']))
        # print(requirements)
        if requirements:
            self.vstore.add_requirements(requirements)
        return {"user_requirements":requirements}
    
    def write_user_stories(self, state: State) -> dict:
        """
        Function to create user stories
        """

        print("\n\n i am showing you the list of user stories \n\n")
        

        System_prompt = f"""
                    You are an expert Agile coach specializing in writing high-quality user stories.  
                    Your task is to convert given software requirements into well-structured user stories  
                    following the format:  

                    **As a [user role], I want to [goal], so that [benefit].**  

                    Ensure clarity, correctness, and alignment with Agile best practices.  
                    If a requirement is vague, assume reasonable details to create a meaningful user story.  

                    **Do not add thought process, analysis. simply output user stories.**
                    output the user stories in a list format.
                """
        user_prompt = f"""
                        Here are the system requirements: {state["user_requirements"]}
                        Convert these into user stories.
                    """
        user_stories = self.llm.invoke(System_prompt + user_prompt)
        
        # print(user_stories, "\n")
        if user_stories:
            # unpack_us = SDLCNode.unpack_messages(user_stories)
            unpack_us = user_stories.content
            # print(unpack_us)
            us_list = unpack_us.split("\n")
            self.vstore.add_ustories(us_list)
        return {'user_stories': user_stories}

