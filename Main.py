import streamlit as st
import redirect as rd

import os
import tempfile
import time

from llama_index import StorageContext, LLMPredictor
from llama_index import TreeIndex, load_index_from_storage
from llama_index import ServiceContext
from langchain.prompts import StringPromptTemplate
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
from langchain.agents import AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain import LLMChain, OpenAI
from llama_index.indices.tree.tree_root_retriever import TreeRootRetriever
import re
from langchain.chat_models import ChatOpenAI
from llama_index.tools import QueryEngineTool, ToolMetadata
from llama_index.query_engine import SubQuestionQueryEngine
from langchain.agents import Tool
from llama_index.query_engine import RetrieverQueryEngine
import openai
# import nest_asyncio

# nest_asyncio.apply()

def call_openai_api(*args, **kwargs):
    return openai.ChatCompletion.create(*args, **kwargs)

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai.api_key = os.getenv("OPENAI_API_KEY")

query_engine_tools = []

import asyncio
def get_or_create_eventloop():
    try:
        return asyncio.get_event_loop()
    except RuntimeError as ex:
        if "There is no current event loop in thread" in str(ex):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return asyncio.get_event_loop()

def remove_formatting(output):
    output = re.sub('\[[0-9;m]+', '', output)  
    output = re.sub('\', '', output) 
    return output.strip()

@st.cache_resource
def preprocessing():
    names = ["The Insurance Act, 1938: Regulations and Restrictions for Insurance Companies in India", "The IRDA Act and Insurance Laws (Amendment) Act 2015", "Motor Vehicle Third-Party Insurance Base Premiums and Liabilities Rules, 2022", "Insurance Business in Special Economic Zone: Exemptions and Regulations"]
    descriptions = ["The go-to document for Insurance Rules. The Insurance Act, 1938 is an Act to consolidate and amend the law relating to the business of insurance in India. It outlines the regulations for insurance companies, including registration, capital requirements, investment, loans and management, investigation, appointment of staff, control over management, amalgamation and transfer of insurance business, commission and rebates, licensing of agents, management by administrator, and acquisition of the undertakings of insurers in certain cases. It also outlines the voting rights of shareholders, the requirements for making a declaration of interest in a share held in the name of another person, the requirements for the separation of accounts and funds for different classes of insurance business, the audit and actuarial report and abstract that must be conducted annually, the power of the Authority to order revaluation and to inspect returns, the power of the Authority to make rules and regulations, the power of the Authority to remove managerial persons from office, appoint additional directors, and issue directions regarding re-insurance treaties, the power of the Authority to enter and search any building or place where books, accounts, or other documents relating to any claim, rebate, or commission are kept, the prohibition of cessation of payments of commission, the prohibition of offering of rebates as an inducement to take out or renew an insurance policy, the process for issuing a registration to act as an intermediary or insurance intermediary, the process for repudiating a life insurance policy on the ground of fraud, the prohibition of insurance agents, intermediaries, or insurance intermediaries to be or remain a director in an insurance company, the requirement to give notice to the policy-holder informing them of the options available to them on the lapsing of a policy, and the power of the National Company Law Tribunal to order the winding up of an insurance company. Penalties for non-compliance range from fines to imprisonment. The Act also outlines the formation of the Life Insurance Council and General Insurance Council, and the Executive Committees of each, the Tariff Advisory Committee, and the obligations of insurers in respect of rural or social or unorganized sector and backward classes.", "The go-to document for information on the Insurance Authority. This document outlines the Insurance Regulatory and Development Authority (IRDA) Act, 1999, which was amended by the Insurance Laws (Amendment) Act, 2015. The Act provides protection of action taken in good faith for the Central Government, officers, members, and other employees of the Authority. It also allows the Authority to delegate powers and functions to the Chairperson or other members or officers of the Authority, and to form committees of members. The amendments include the substitution of \"Authority\" for \"Controller\" in most cases, the introduction of an Indian insurance company, the introduction of a requirement for a paid-up equity capital of Rs. 100 crores for life or general insurance, and the introduction of a requirement for a paid-up equity capital of Rs. 200 crores for re-insurance. It also includes changes to sections 2B, 2C, 6, 70, 95, 101A, 101B, 102-105, 110H, 114, 114A, and 24A of the IRDA Act.", "This document outlines the Motor Vehicles (Third Party Insurance Base Premium and Liability) Rules, 2022, published by the Ministry of Road Transport and Highways. It provides base premiums and liabilities for various categories of motor vehicles, including motorised two wheelers, motorised three wheelers, motorised pedal cycles, four wheeled vehicles, trailers, agricultural tractors, and special types of vehicles. The rates for each type of vehicle vary depending on the size and type of vehicle, and discounts are available for vintage cars and hybrid electric vehicles. The rules also provide long-term rates for three years single premium.", "This document is a notification from the Ministry of Finance in India, issued on March 27, 2015. It states that certain sections of the Insurance Act, 1938 (4 of 1938) will not apply to Indian Insurance Companies, insurance co-operative societies, and certain body corporates carrying out insurance business in Special Economic Zones. The sections that will not apply are listed in the notification."]
    temp = ['insurance', 'irda', 'mv', 'sez']
    for n, x in enumerate(temp):
        storage_context = StorageContext.from_defaults(
            persist_dir = x,
        )
        index = load_index_from_storage(storage_context)
        engine = index.as_query_engine(similarity_top_k = 3)
        query_engine_tools.append(QueryEngineTool(
            query_engine = engine,
            metadata = ToolMetadata(name = names[n], description = descriptions[n])
        ))
    st.header('Document Headings and Descriptions -')

    for i in range(4):
        st.subheader(f"{i + 1}) " + names[i])
        st.write(descriptions[i])

    s_engine = SubQuestionQueryEngine.from_defaults(query_engine_tools = query_engine_tools)

    tools = [Tool(
        name = "Llama-Index",
        func = s_engine.query,
        description = f"Useful for when you want to answer questions. The input to this tool should be a complete English sentence. Works best if you redirect the entire query back into this.",
        return_direct = True
        )
    ]

    template1 = """You are a Smart Insurance Agent Assistant. The Agent will ask you domain specific questions. The tools provided to you have smart interpretibility if you specify keywords in your query to the tool [Example a query for two wheeler insurance rules should mention two wheelers]. You have access to the following tools:

                {tools}

                Use the following format:

                Question: the input question you must answer
                Thought: you should always think about what to do
                Action: the action to take, should be one of [{tool_names}]
                Action Input: the input to the action
                Observation: the result of the action
                ... (this Thought/Action/Action Input/Observation can repeat N times)
                Thought: I now know the final answer
                Final Answer: the final answer to the original input question

                Begin! Remember to be ethical and articulate when giving your final answer. Use lots of "Arg"s

                Question: {input}
                {agent_scratchpad}"""

    prompt = CustomPromptTemplate(
        template = template1,
        tools = tools,
        input_variables=["input", "intermediate_steps"]
    )

    output_parser = CustomOutputParser()

    llm = OpenAI(temperature = 0)
    llm_chain = LLMChain(llm = llm, prompt = prompt)

    tool_names = [tool.name for tool in tools]
    agent = LLMSingleActionAgent(
        llm_chain = llm_chain, 
        output_parser = output_parser,
        stop = ["\nObservation:"], 
        allowed_tools = tool_names
    )

    agent_chain = AgentExecutor.from_agent_and_tools(tools = tools, agent = agent, verbose = True)

    return agent_chain
    
@st.cache_resource
def run(query):
    if query:
        with rd.stdout() as out:
            ox = agent_chain.run(query)
        output = out.getvalue()
        output = remove_formatting(output)
        st.write(ox.response)
        return True

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        intermediate_steps = kwargs.pop("intermediate_steps")
        thoughts = ""
        for action, observation in intermediate_steps:
            thoughts += action.log
            thoughts += f"\nObservation: {observation}\nThought: "
        kwargs["agent_scratchpad"] = thoughts
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)
    
class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        if "Final Answer:" in llm_output:
            return AgentFinish(
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2)
        return AgentAction(tool=action, tool_input=action_input.strip(" ").strip('"'), log=llm_output)

st.set_page_config(layout = "wide")

st.title("Tools and Subqueries using LangChain and Llama-Index")
# st.markdown('_The headings and descriptions given below are generated using LLMs._')

llm_predictor = LLMPredictor(llm = ChatOpenAI(temperature = 0, model_name = 'gpt-3.5-turbo', max_tokens = -1))

storage_context = StorageContext.from_defaults()
service_context = ServiceContext.from_defaults(llm_predictor = llm_predictor)

agent_chain = preprocessing()
ack = False

if agent_chain:
    query = st.text_input('Enter your Query.', key = 'query_input')
    ack = run(query)
    if ack:
        ack = False
        query = st.text_input('Enter your Query.', key = 'new_query_input')
        ack = run(query)
        if ack:
            ack = False
            query = st.text_input('Enter your Query.', key = 'new_query_input1')
            ack = run(query)
            if ack:
                ack = False
                query = st.text_input('Enter your Query.', key = 'new_query_input2')
                ack = run(query)
                if ack:
                    ack = False
                    query = st.text_input('Enter your Query.', key = 'new_query_input3')
                    ack = run(query)
                    if ack:
                        ack = False
                        query = st.text_input('Enter your Query.', key = 'new_query_input4')
                        ack = run(query)
                        if ack:
                            ack = False
                            query = st.text_input('Enter your Query.', key = 'new_query_input5')
                            ack = run(query)
        