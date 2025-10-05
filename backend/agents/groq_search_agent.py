# agents/groq_search_agent.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import os
import json
from dotenv import load_dotenv
import uvicorn
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

# Load environment variables from a .env file
load_dotenv()

# --- Configuration & API Key Check ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY is not set in the environment. The agent cannot start.")

# --- LLM Initialization ---
# Use ChatGroq from langchain_groq for a consistent interface
# The 'compound-beta' model is specifically designed for search-augmented responses.
# Note: The parameter is 'model_name' for ChatGroq, but we pass it as 'model' in model_kwargs
# for models like compound-beta that are not standard chat models.
# For tool-use models, additional_kwargs is the correct way to get tool call info.
llm = ChatGroq(model="groq/compound")


# --- Agent Definition ---
AGENT_DEFINITION = {
  "id": "groq_search_agent",
  "owner_id": "orbimesh-vendor",
  "name": "Orbimesh Groq Compound Agent",
  "description": "A powerful research agent that uses Groq's `compound-beta` model to perform real-time web searches and provide summarized, cited answers to user queries.",
  "capabilities": [
    "perform web search and summarize"
  ],
  "price_per_call_usd": 0.005,
  "status": "active",
  "public_key_pem": "-----BEGIN PUBLIC KEY-----\nMCowBQYDK2VwAyEA3FcU8hPhmFLgez6qPf801aQahasAlG5S4MPb16nWJPA=\n-----END PUBLIC KEY-----",
  "endpoints": [
    {
      "endpoint": "http://localhost:8050/search-and-summarize",
      "http_method": "POST",
      "description": "Performs a real-time web search using Groq's `compound-beta` system and returns a synthesized answer with citations. Supports domain filtering.",
      "parameters": [
        {
          "name": "query",
          "param_type": "string",
          "required": True,
          "description": "The question or topic to search for and summarize (e.g., 'What are the latest developments in AI?')."
        },
        {
          "name": "include_domains",
          "param_type": "array[string]",
          "required": False,
          "description": "A list of domains to restrict the search to (e.g., ['techcrunch.com', '*.gov'])."
        },
        {
          "name": "exclude_domains",
          "param_type": "array[string]",
          "required": False,
          "description": "A list of domains to exclude from the search (e.g., ['wikipedia.org'])."
        }
      ]
    }
  ]
}

app = FastAPI(title="Groq Compound Search Agent")

# --- Pydantic Models ---
class SearchSettings(BaseModel):
    include_domains: Optional[List[str]] = None
    exclude_domains: Optional[List[str]] = None

class SearchRequest(BaseModel):
    query: str
    search_settings: Optional[SearchSettings] = None

class SearchResponse(BaseModel):
    answer: str
    sources: Optional[List[dict]] = None

# --- API Endpoints ---
@app.get("/")
def read_root():
    return AGENT_DEFINITION

@app.post("/search-and-summarize", response_model=SearchResponse)
async def get_search_summary(req: SearchRequest):
    """
    Takes a user query, instructs the Groq model to search the web,
    and returns a summarized answer with sources using ChatGroq.
    """
    try:
        # Construct model-specific arguments for tool use
        model_kwargs = {"model": "compound-beta"}
        if req.search_settings:
            model_kwargs["search_settings"] = req.search_settings.model_dump(exclude_none=True)

        # Bind the kwargs to the LLM instance
        bound_llm = llm.bind(**model_kwargs)
        
        # Invoke the model with the user's query
        response = await bound_llm.ainvoke([HumanMessage(content=req.query)])
        
        answer = response.content
        if not isinstance(answer, str):
            answer = str(answer) # Ensure content is a string

        # Extract search results from tool calls in the response metadata
        sources = []
        if response.additional_kwargs and "tool_calls" in response.additional_kwargs:
            for tool_call in response.additional_kwargs["tool_calls"]:
                if tool_call.get("function", {}).get("name") == "search":
                    try:
                        # The output from the tool is a string that needs to be parsed as JSON
                        output_str = tool_call.get("function", {}).get("output", "[]")
                        output = json.loads(output_str)
                        sources.extend(output)
                    except json.JSONDecodeError:
                        # Handle cases where the output is not valid JSON, gracefully skip
                        pass

        return SearchResponse(answer=answer, sources=sources)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the search: {e}")

if __name__ == "__main__":
    uvicorn.run("groq_search_agent:app", host="127.0.0.1", port=8050, reload=True)