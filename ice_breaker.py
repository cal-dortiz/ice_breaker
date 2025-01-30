from langchain_core.prompts import PromptTemplate
# from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain_core. output_parsers import StrOutputParser
from third_parties.linkedin import scrape_linkedin_profile
from agents.linkedin_lookup_agent import lookup as linkedin_lookup_agent
from dotenv import load_dotenv

load_dotenv()

def ice_break_with(name: str) -> str:
    linkedin_url = linkedin_lookup_agent(name=name)
    linkedin_data = scrape_linkedin_profile(linkedin_profile_url=linkedin_url)

    summary_template = """
        Given the linkedin information {information} about a personf from I want you to create:
        1. a short summary
        2. two interestng facts about them
    """

    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )

    # llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")
    llm = ChatOllama(model="llama3.2")

    chain = summary_prompt_template | llm 

    res = chain.invoke(input={"information": linkedin_data})

    print(res)

if __name__ == "__main__":
    print("Ice Breaker Enter")
    ice_break_with(name="")



