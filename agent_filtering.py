# agent_filtering.py

import json
from typing import List
from agents import call_perplexity_search, call_groq_chatbot

def find_relevant_competitors(
    candidates: List[str],
    input_company: str,
    search_complexity: str = 'medium'
) -> List[str]:
    """
    Filters a list of potential competitors down to the 3–10 most relevant ones.
    
    1) A web search agent (Perplexity) retrieves background information for each company.
    2) A chat agent (Groq) selects the most relevant companies based on this information.
    
    Args:
        candidates: List of all potential competitor names.
        input_company: Name of the input company.
        search_complexity: 'low' | 'medium' | 'high' for the web search.
        
    Returns:
        A list of the 3–10 most relevant competitors.
    """
    # 1) Gather background information
    system_prompt1 = (
        "You are a research agent that gathers background information "
        "for each company in a list. For each company, extract key facts: "
        "industry, company size (employees or revenue), headquarters location, "
        "and any recent notable news."
    )
    user_prompt1 = (
        f"Input company: {input_company}\n"
        "Candidate companies:\n" +
        "\n".join(f"- {c}" for c in candidates) +
        "\n\nPlease return a concise, bullet-point summary per company."
    )
    info_resp = call_perplexity_search(
        system_prompt=system_prompt1,
        user_prompt=user_prompt1,
        complexity=search_complexity
    )
    background_info = info_resp['content']

    # 2) Select the most relevant competitors
    system_prompt2 = (
        "You are an expert competitive analyst. "
        "Given a list of candidate companies and their background information, "
        f"select the 3 to 10 most relevant competitors for {input_company}. "
        "Respond *only* with a JSON array of the selected company names."
    )
    user_prompt2 = (
        "Candidate companies:\n" +
        "\n".join(f"- {c}" for c in candidates) +
        "\n\nBackground information:\n" + background_info +
        "\n\nOutput the JSON array now."
    )
    selection_resp = call_groq_chatbot(system_prompt2, user_prompt2)

    # Attempt to parse JSON
    try:
        selected = json.loads(selection_resp)
        if isinstance(selected, list):
            return selected
    except json.JSONDecodeError:
        pass

    # Fallback: Lines without JSON
    lines = [
        line.strip("- ").strip()
        for line in selection_resp.splitlines()
        if line.strip()
    ]
    return lines[:10]
