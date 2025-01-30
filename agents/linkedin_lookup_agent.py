import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool
from langchain.agents import (
    create_react_agent,
    AgentExecutor
)
from tools.tools import get_profile_url_tavily

from langchain import hub

load_dotenv()


def lookup(name:str):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    template = """
        Given the full name of {name_of_person} I want you to get me a link to their LinkedIn profile page.
        Your answer should only contain a URL
    """

    prompt_template = PromptTemplate(
        template=template,
        input_variables=["name_of_person"]
    )

    tools_for_agent = [
        Tool(
            name="Crawl Google for LinkedIn profile page",
            func = get_profile_url_tavily,
            description="Useful for when you need to get the LinkedIn page URL"
        )
    ]

    react_prompt = hub.pull("hwchase17/react")

    agent = create_react_agent(
        llm=llm,
        tools=tools_for_agent,
        prompt=react_prompt
    )

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_for_agent,
        verbose=True
    )

    results = agent_executor.invoke(
        input={"input":prompt_template.format_prompt(name_of_person=name)}
    )

    linkedin_profile_url = results['output']

    return "https://www.linkedin.com"