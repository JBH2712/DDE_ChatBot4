import os
import pandas as pd
from openai import OpenAI

# Load API keys from local files into environment variables
with open(r'Sonar_API.txt', 'r', encoding='utf-8') as f:
    os.environ['SONAR_API_KEY'] = f.readline().strip()

with open(r'Groq_API.txt', 'r', encoding='utf-8') as f:
    os.environ['GROQ_API_KEY'] = f.readline().strip()

with open(r'OpenAI_API.txt', 'r', encoding='utf-8') as f:
    os.environ['OPENAI_API_KEY'] = f.readline().strip()

# Initialize OpenAI client after loading API key
client = OpenAI(api_key=os.environ.get('OPENAI_API_KEY'))

import os
import requests
from typing import List, Dict, Any

def call_perplexity_search(
    system_prompt: str,
    user_prompt: str,
    complexity: str = 'low',
    top_k: int = 0,
    max_tokens: int = 15000,
    temperature: float = 0.2,
    top_p: float = 0.9,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 1.0,
    api_key_env: str = 'SONAR_API_KEY'
) -> Dict[str, Any]:
    """
    Makes a request to the Perplexity AI (Sonar) API.

    Args:
        system_prompt (str): The system message for the agent.
        user_prompt (str): The user message (your search query).
        complexity (str): Search complexity: 'low', 'medium', or 'high'.
        top_k (int): Number of related questions (related questions).
        max_tokens (int): Maximum number of tokens in the response.
        temperature (float): Sampling temperature.
        top_p (float): Nucleus sampling parameter.
        presence_penalty (float): Penalty for new tokens.
        frequency_penalty (float): Penalty for repeated tokens.
        api_key_env (str): Name of the environment variable containing the API key.

    Returns:
        dict: {
            'content': str,           # Generated answer
            'citations': List[str],   # Sources
            'raw_response': Any       # Original response (JSON or text)
        }
    """
    url = "https://api.perplexity.ai/chat/completions"
    api_key = os.getenv('SONAR_API_KEY')
    if not api_key:
        raise EnvironmentError(f"Please set the environment variable '{api_key_env}'.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "sonar",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "presence_penalty": presence_penalty,
        "frequency_penalty": frequency_penalty,
        "return_images": False,
        "return_related_questions": False,
        "top_k": top_k,
        "stream": False,
        "web_search_options": {"search_context_size": complexity}
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code != 200:
        return {
            'content': f"Error {response.status_code}: {response.text}",
            'citations': [],
            'raw_response': response.text
        }

    data = response.json()
    choice = data.get('choices', [{}])[0].get('message', {})
    return {
        'content': choice.get('content', ''),
        'citations': data.get('citations', []),
        'raw_response': data
    }


import os
import json
from typing import Dict, Any
from langchain_groq import ChatGroq
from langchain.schema import SystemMessage, HumanMessage

def call_groq_chatbot(
    system_prompt: str,
    user_prompt: str,
    model: str = "llama-3.3-70b-versatile",
    temperature: float = 0.0,
    max_tokens: int = 8192,
    api_key_env: str = "GROQ_API_KEY"
) -> str:
    """
    Wrapper for Groq Cloud Chat with system and user prompts.

    Args:
        system_prompt (str): Instructions to the model (system message).
        user_prompt (str): The actual user query (human message).
        model (str): Model name, e.g., "llama3-8b-8192".
        temperature (float): Sampling temperature (0.0–1.0).
        max_tokens (int): Maximum number of tokens in the response.
        api_key_env (str): Name of the environment variable containing the API key.

    Returns:
        str: Generated response from the Groq LLM.
    """
    api_key = os.environ['GROQ_API_KEY']
    if not api_key:
        raise EnvironmentError(f"Please set the environment variable '{api_key_env}'.")

    llm = ChatGroq(
        api_key=api_key,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens
    )

    # Create message list with system and human prompts
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    # Execute the chat
    response = llm(messages)

    return response.content


import os
from typing import Iterator, Dict, List, Optional
import pandas as pd
from openai import OpenAI, OpenAIError


class OpenAIStreamingAgent:
    """
    Base agent for streaming responses via OpenAI ChatCompletion.
    """
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key_env: str = "OPENAI_API_KEY"
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise EnvironmentError(f"Please set the environment variable '{api_key_env}'.")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.history: List[Dict[str, str]] = []

    def add_system_message(self, content: str) -> None:
        """
        Add a system message to the conversation history.
        """
        self.history.append({"role": "system", "content": content})

    def add_user_message(self, content: str) -> None:
        """
        Add a user message to the conversation history.
        """
        self.history.append({"role": "user", "content": content})

    def stream_response(self) -> Iterator[str]:
        """
        Fallback: ChatCompletion streaming response.
        """
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=self.temperature,
                stream=True
            )
        except OpenAIError as e:
            raise RuntimeError(f"OpenAI error: {e}")
        
        assistant_message = ""
        for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                assistant_message += token
                yield token
        self.history.append({"role": "assistant", "content": assistant_message})

    def reset_history(self) -> None:
        """
        Reset the conversation history.
        """
        self.history = []


class CompetitiveChatAgent(OpenAIStreamingAgent):
    """
    Extended chat agent with access to a DataFrame, web search, and file search tools.
    Optionally uses streaming via the Responses API.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        vector_store_id: Optional[str] = "vs_68102a9708508191a9b9e7396475f7cd",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        api_key_env: str = "OPENAI_API_KEY",
        country: str = "PL"
    ):
        super().__init__(model=model, temperature=temperature, api_key_env=api_key_env)
        self.df = df
        self.vector_store_id = vector_store_id
        self.country = country

    def get_company_info(self, company_name: str) -> str:
        row = self.df[self.df["Company name Latin alphabet"] == company_name]
        if row.empty:
            return "No data found."
        return row.iloc[0].get("Description", "No description available.")

    def init_conversation(
        self,
        input_company: str,
        comp_description: str,
        competitors: List[str],
        background_info: str
    ) -> None:
        """
        Builds the system prompt where the internal tool usage steps are not
        disclosed, only the relevant data is.
        """
        sys = (
            f"You are a competitive analyst bot.\n"
            f"Input company: {input_company}\n"
            f"Company description: {comp_description}\n\n"
            f"Relevant competitors: {', '.join(competitors)}\n"
            f"Background information about competitors:\n{background_info}\n\n"
            "When the user asks a question, answer it as best as possible "
            "based on all of the above information. "
            "Never mention how you obtained the information."
        )
        self.add_system_message(sys)

    from openai import OpenAIError

    def handle_user_with_tools(
        self,
        user_input: str,
        max_output_tokens: int = 2048,
        top_p: float = 1.0
    ) -> str:
        """
        Builds the history, calls the full response via the Responses API
        (with file_search + web_search_preview) and extracts only the
        text part from response.output.
        """
        # 1) Save user message
        self.add_user_message(user_input)

        # 2) Tools list (file_search remains included)
        tools = []
        if self.vector_store_id:
            tools.append({
                "type": "file_search",
                "vector_store_ids": [self.vector_store_id]
            })
        tools.append({
            "type": "web_search_preview",
            "user_location": {"type": "approximate", "country": self.country},
            "search_context_size": "medium"
        })

        # 3) Responses API call (stream=False)
        try:
            response = self.client.responses.create(
                model=self.model,
                input=self.history,
                text={"format": {"type": "text"}},
                reasoning={},
                tools=tools,
                temperature=self.temperature,
                max_output_tokens=max_output_tokens,
                top_p=top_p,
                stream=False,
                store=True
            )
        except OpenAIError as e:
            raise RuntimeError(f"Responses API error: {e}")

        # 4) Assemble text from response.output
        full_text = ""
        for out in getattr(response, "output", []):
            # We skip tool invocations and only look at message outputs
            if hasattr(out, "content"):
                # out.content is a list of ResponseOutputText objects
                for txt in out.content:
                    # Each has a .text attribute with the desired string
                    full_text += txt.text

        # 5) If no text found, fallback to response.text.format
        if not full_text:
            # Some versions might pack the result again under response.text ...
            full_text = getattr(response, "text", {}).get("type", "")  # adjust if necessary

        # 6) Save to history and return
        self.history.append({"role": "assistant", "content": full_text})
        return full_text


