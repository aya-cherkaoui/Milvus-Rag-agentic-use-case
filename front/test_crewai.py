from helpers.utils import init_env_from_yaml
from helpers.watsonx_caller import WxaiWrapper
import sys
import time
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain.agents import Tool
from ibm_watsonx_ai.foundation_models import Model
from ibm_watsonx_ai.credentials import Credentials
from langchain.llms.base import LLM
from langchain.agents import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from typing import Any, List, Mapping, Optional
import os
import re


# Initialiser l'environnement à partir d'un fichier YAML
init_env_from_yaml()


#print(os.environ['WATSONX_APIKEY'])
#print(os.environ["WATSONX_URL"])
#print(os.environ["WATSONX_PROJECT_ID"])
WATSONX_URL="https://us-south.ml.cloud.ibm.com/"
WATSONX_PROJECT_ID= "a71069b6-7658-4eda-be21-6c22f7e8fc29"
WATSONX_APIKEY = "wv4PNVaimuKg7Kb5oLCRzKpcFqbwtUEHr6PPP4-t-zza"
#WATSONX_LLAMA3_MODEL_ID = "watsonx/mistralai/mistral-large"
WATSONX_LLAMA3_MODEL_ID="watsonx/meta-llama/llama-3-2-3b-instruct"


model_id = 'meta-llama/llama-3-2-3b-instruct'
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 500,
    "temperature": 0.7,
    "top_k": 50,
    "top_p": 1,
    "repetition_penalty": 1
}
credentials = Credentials(url=WATSONX_URL, api_key=WATSONX_APIKEY)

ibm_model = Model(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id=WATSONX_PROJECT_ID
)

class IBMWatsonLLM(LLM):
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        return ibm_model.generate_text(prompt=prompt, guardrails=False)

    @property
    def _llm_type(self) -> str:
        return "ibm_watson"

    def generate_text(self, prompt):
        return self._call(prompt)
llm = IBMWatsonLLM()

"""
llm = LLM(
    model=WATSONX_LLAMA3_MODEL_ID,
    base_url=WATSONX_URL,
    project_id=WATSONX_PROJECT_ID,
    api_key=WATSONX_APIKEY,
    max_tokens=2000,
)
"""
google_search = GoogleSearchAPIWrapper()

google_search_tool = Tool(
    name="google_search",
    func=google_search.run,
    description="Search the web via Google"
)

#to keep track of tasks performed by agents
task_values = []

def create_crewai_setup(product_name):
    # Define Agents
    market_research_analyst = Agent(
        role="Market Research Analyst",
        goal=f"""Analyze the market demand for {product_name} and 
                 suggest marketing strategies""",
        backstory=f"""Expert at understanding market demand, target audience, 
                      and competition for products like {product_name}. 
                      Skilled in developing marketing strategies 
                      to reach a wide audience.""",
        verbose=True,
        allow_delegation=True,
        tools=[google_search_tool],
        llm=llm,
    )

    technology_expert = Agent(
        role="Technology Expert",
        goal=f"Assess technological feasibilities and requirements for producing high-quality {product_name}",
        backstory=f"""Visionary in current and emerging technological trends, 
                      especially in products like {product_name}. 
                      Identifies which technologies are best suited 
                      for different business models.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )

    business_consultant = Agent(
        role="Business Development Consultant",
        goal=f"""Evaluate the business model for {product_name}, 
               focusing on scalability and revenue streams""",
        backstory=f"""Seasoned in shaping business strategies for products like {product_name}. 
                      Understands scalability and potential 
                      revenue streams to ensure long-term sustainability.""",
        verbose=True,
        allow_delegation=True,
        llm=llm,
    )


    # Define Tasks
    task1 = Task(
        description=f"""Analyze the market demand for {product_name}. Current month is November 2024.
                        Write a report on the ideal customer profile and marketing 
                        strategies to reach the widest possible audience. 
                        Include at least 10 bullet points addressing key marketing areas.""",
        expected_output="Report on market demand analysis and marketing strategies.",
        agent=market_research_analyst,
    )
    # Define Task 2
    task2 = Task(
        description=f"""Assess the technological aspects of manufacturing 
                    high-quality {product_name}. Write a report detailing necessary 
                    technologies and manufacturing approaches. 
                    Include at least 10 bullet points on key technological areas.""",
        expected_output="Report on technological aspects of manufacturing.",
        agent=technology_expert,
    )
    # Define Task 3
    task3 = Task(
        description=f"""Summarize the market and technological reports 
                    and evaluate the business model for {product_name}. 
                    Write a report on the scalability and revenue streams 
                    for the product. Include at least 10 bullet points 
                    on key business areas. Give Business Plan, 
                    Goals and Timeline for the product launch. Current month is Jan 2024.""",
        expected_output="Report on business model evaluation and product launch plan.",
        agent=business_consultant,
    )

    # Create and Run the Crew
    product_crew = Crew(
        agents=[market_research_analyst, technology_expert, business_consultant],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential,
    )



    crew_result = product_crew.kickoff()
    return crew_result

#display the console processing on streamlit UI
class StreamToExpander:
    def __init__(self, expander):
        self.expander = expander
        self.buffer = []
        self.colors = ['red', 'green', 'blue', 'orange']  # Define a list of colors
        self.color_index = 0  # Initialize color index

    def write(self, data):
        # Filter out ANSI escape codes using a regular expression
        cleaned_data = re.sub(r'\x1B\[[0-9;]*[mK]', '', data)

        # Check if the data contains 'task' information
        task_match_object = re.search(r'\"task\"\s*:\s*\"(.*?)\"', cleaned_data, re.IGNORECASE)
        task_match_input = re.search(r'task\s*:\s*([^\n]*)', cleaned_data, re.IGNORECASE)
        task_value = None
        if task_match_object:
            task_value = task_match_object.group(1)
        elif task_match_input:
            task_value = task_match_input.group(1).strip()

        if task_value:
            st.toast(":robot_face: " + task_value)

        # Check if the text contains the specified phrase and apply color
        if "Entering new CrewAgentExecutor chain" in cleaned_data:
            # Apply different color and switch color index
            self.color_index = (self.color_index + 1) % len(self.colors)  # Increment color index and wrap around if necessary

            cleaned_data = cleaned_data.replace("Entering new CrewAgentExecutor chain", f":{self.colors[self.color_index]}[Entering new CrewAgentExecutor chain]")

        if "Market Research Analyst" in cleaned_data:
            # Apply different color 
            cleaned_data = cleaned_data.replace("Market Research Analyst", f":{self.colors[self.color_index]}[Market Research Analyst]")
        if "Business Development Consultant" in cleaned_data:
            cleaned_data = cleaned_data.replace("Business Development Consultant", f":{self.colors[self.color_index]}[Business Development Consultant]")
        if "Technology Expert" in cleaned_data:
            cleaned_data = cleaned_data.replace("Technology Expert", f":{self.colors[self.color_index]}[Technology Expert]")
        if "Finished chain." in cleaned_data:
            cleaned_data = cleaned_data.replace("Finished chain.", f":{self.colors[self.color_index]}[Finished chain.]")

        self.buffer.append(cleaned_data)
        if "\n" in data:
            self.expander.markdown(''.join(self.buffer), unsafe_allow_html=True)
            self.buffer = []

# Streamlit interface
def run_crewai_app():
    st.title("Watsonx AI Agent for product potentials")
    with st.expander("About the Team:"):
        st.subheader("Diagram")
        left_co, cent_co,last_co = st.columns(3)

        st.subheader("Market Research Analyst")
        st.text("""       
        Role = Market Research Analyst
        Goal = Analyze the market demand for {product_name} and suggest marketing strategies
        Backstory = Expert at understanding market demand, target audience, 
                    and competition for products like {product_name}. 
                    Skilled in developing marketing strategies 
                    to reach a wide audience.
        Task = Analyze the market demand for {product_name}. Current month is Jan 2024.
               Write a report on the ideal customer profile and marketing 
               strategies to reach the widest possible audience. 
               Include at least 10 bullet points addressing key marketing areas. """)
        
        st.subheader("Technology Expert")
        st.text("""       
        Role = Technology Expert
        Goal = Assess technological feasibilities and requirements for producing high-quality {product_name}
        Backstory = Visionary in current and emerging technological trends, 
                    especially in products like {product_name}. 
                    Identifies which technologies are best suited 
                    for different business models. 
        Task = Assess the technological aspects of manufacturing 
               high-quality {product_name}. Write a report detailing necessary 
               technologies and manufacturing approaches. 
               Include at least 10 bullet points on key technological areas.""")

        st.subheader("Business Development Consultant")
        st.text("""       
        Role = Business Development Consultant 
        Goal= Evaluate the business model for {product_name}
              focusing on scalability and revenue streams
        Backstory = Seasoned in shaping business strategies for products like {product_name}. 
                    Understands scalability and potential 
                    revenue streams to ensure long-term sustainability.
        Task = Summarize the market and technological reports 
               and evaluate the business model for {product_name}. 
               Write a report on the scalability and revenue streams 
               for the product. Include at least 10 bullet points 
               on key business areas. Give Business Plan, 
               Goals and Timeline for the product launch. Current month is Jan 2024. """)
    
    product_name = st.text_input("Enter a product name to analyze the market and business strategy.")

    if st.button("Run Analysis"):
        # Placeholder for stopwatch
        stopwatch_placeholder = st.empty()
        
        # Start the stopwatch
        start_time = time.time()
        with st.expander("Processing!"):
            sys.stdout = StreamToExpander(st)
            with st.spinner("Generating Results"):
                crew_result = create_crewai_setup(product_name)

        # Stop the stopwatch
        end_time = time.time()
        total_time = end_time - start_time
        stopwatch_placeholder.text(f"Total Time Elapsed: {total_time:.2f} seconds")

        st.header("Tasks:")
        st.table({"Tasks" : task_values})

        st.header("Results:")
        st.markdown(crew_result)

if __name__ == "__main__":
    run_crewai_app()