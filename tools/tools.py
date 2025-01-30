from langchain_community.tools.tavily_search import TavilySearchResults

def get_profile_url_tavily(name: str):
    """Searches for linkedin or twitter profile page"""

    search = TavilySearchResults()
    results = search.run(f"{name}")
    return results