from src.langgraphagenticai.state.state import State

class HealthChatbotWithToolNode:
    """
    Chatbot logic enhanced with tool integration.
    """
    def __init__(self,model, session_data):
        self.llm = model
        self.session_data = session_data

    def process(self, state: State) -> dict:
        """
        Processes the input state and generates a response with tool integration.
        """

        System_message = "Hello, I'm a digital assistant that can help schedule the first study appointment for you."
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])

        return {"messages": llm_response}
    
    # def create_chatbot(self, tools):
    #     """
    #     Returns a chatbot node function.
    #     """
    #     llm_with_tools = self.llm.bind_tools(tools)

    #     def chatbot_node(state: State):
    #         """
    #         Chatbot logic for processing the input state and returning a response.
    #         """
    #         System_message = "Hello, I'm a digital assistant that can help you schedule the first study appointment for you."
    #         return {"messages": [llm_with_tools.invoke(System_message+state["messages"])]}

    #     return chatbot_node
 
    def prehealthcheck_gudielines(self) -> dict:
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
        self.session_data['input_required'] = True
        return {'messages':guideline}
    
    def calendar(self) ->dict:
        """
        Returns the list of available dates for the appointment
        """
        dates = ['2-MAY-2025', '10-MAY-2025', '31-MAY-2025']
        self.session_data['date_decision'] = dates
        return {'messages':dates}

    def site_locations(self) -> dict:
        """
        Returns list of locations available at for the patient
        """
        sites = ['Hyderabad', 'Bangalore']
        self.session_data['site_decision'] = sites
        return  {'messages':sites}

    def appoint_confirmation(self):
        """
        shows appointment confirmation
        """
        conf = f"""
                your appointment is confirmed at {self.session_data['site_decision']}, {self.session_data['date_decision']}
                """
        return {'messages':conf}
        
    
    
    
    
    
