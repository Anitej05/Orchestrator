import type { Agent } from "./types"

// API base URL
const API_BASE_URL = 'http://localhost:8000';

// Fetch all agents from the backend
export async function fetchAllAgents(): Promise<Agent[]> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/agents/all`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error fetching agents from backend:', error);
    throw error;
  }
}

// Fetch agents with filtering options
export async function fetchFilteredAgents(options: {
  maxPrice?: number;
  minRating?: number;
  status?: 'active' | 'inactive';
} = {}): Promise<Agent[]> {
  try {
    const params = new URLSearchParams();
    
    if (options.maxPrice !== undefined) {
      params.append('max_price', options.maxPrice.toString());
    }
    if (options.minRating !== undefined) {
      params.append('min_rating', options.minRating.toString());
    }
    if (options.status) {
      params.append('status_filter', options.status);
    }
    
    const url = `${API_BASE_URL}/api/agents/all${params.toString() ? `?${params.toString()}` : ''}`;
    const response = await fetch(url);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error fetching filtered agents from backend:', error);
    throw error;
  }
}

// Search agents by capabilities
export async function searchAgents(options: {
  capabilities: string[];
  maxPrice?: number;
  minRating?: number;
  similarityThreshold?: number;
}): Promise<Agent[]> {
  try {
    const params = new URLSearchParams();
    
    options.capabilities.forEach(cap => {
      params.append('capabilities', cap);
    });
    
    if (options.maxPrice !== undefined) {
      params.append('max_price', options.maxPrice.toString());
    }
    if (options.minRating !== undefined) {
      params.append('min_rating', options.minRating.toString());
    }
    if (options.similarityThreshold !== undefined) {
      params.append('similarity_threshold', options.similarityThreshold.toString());
    }
    
    const response = await fetch(`${API_BASE_URL}/agents/search?${params.toString()}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agents = await response.json();
    return agents;
  } catch (error) {
    console.error('Error searching agents:', error);
    throw error;
  }
}

// Rate an agent
export async function rateAgent(agentId: string, rating: number): Promise<Agent> {
  try {
    if (rating < 0 || rating > 5) {
      throw new Error('Rating must be between 0 and 5');
    }

    const response = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(agentId)}/rate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(rating),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const updatedAgent = await response.json();
    return updatedAgent;
  } catch (error) {
    console.error('Error rating agent:', error);
    throw error;
  }
}

// Get a specific agent by ID
export async function fetchAgentById(agentId: string): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/${encodeURIComponent(agentId)}`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const agent = await response.json();
    return agent;
  } catch (error) {
    console.error('Error fetching agent by ID:', error);
    throw error;
  }
}

// Register or update an agent
export async function registerAgent(agentData: Agent): Promise<Agent> {
  try {
    const response = await fetch(`${API_BASE_URL}/agents/register`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(agentData),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const registeredAgent = await response.json();
    return registeredAgent;
  } catch (error) {
    console.error('Error registering agent:', error);
    throw error;
  }
}

// Process prompt for orchestration
export async function processPrompt(request: { prompt: string }): Promise<{
  message: string;
  thread_id: string;
  task_agent_pairs: any[];
  final_response?: string;
}> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(request),
    });
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error processing prompt:', error);
    throw error;
  }
}

// Health check
export async function healthCheck(): Promise<{ status: string }> {
  try {
    const response = await fetch(`${API_BASE_URL}/api/health`);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error('Error checking API health:', error);
    throw error;
  }
}

export const frameworks = ["CrewAI", "AutoGen", "LangGraph", "LangChain", "Custom"]


export const capabilities = [
  "find_travel_agent",
  "find_hotel_booking_agent",
  "summarize_documents",
  "write_python_code",
  "research_sales_leads",
  "draft_marketing_emails",
  "Lead Generation",
  "Email Drafting",
  "Translation",
  "Scheduling",
  "Payments",
  "Data Analysis",
  "Content Creation",
  "Research",
  "Customer Support",
  "Social Media Management",
]

// Fallback agents when API is unavailable
export const agents: Agent[] = [
  {
    id: "news_agent",
    owner_id: "orbimesh-vendor",
    name: "Orbimesh News Summary Agent",
    description:
      "Fetches the latest news articles for any topic using Google News RSS, resolves redirect links with Selenium, extracts full text, and generates concise summaries using TextRank.",
    capabilities: [
      "fetch latest news for any query",
      "resolve Google News redirect links",
      "extract full article text",
      "summarize articles with TextRank",
      "get publisher and published date",
    ],
    price_per_call_usd: 0.002,
    status: "active",
    endpoints: [
      {
        endpoint: "http://localhost:8020/news",
        http_method: "GET",
        description:
          "Fetches latest summarized news articles for a given query. Accepts query params 'query', 'count', and 'sentences'. Returns titles, publisher, final URL, publish date, and summary.",
      },
      {
        endpoint: "http://localhost:8020/news",
        http_method: "POST",
        description:
          "Fetches latest summarized news articles for a given query. Accepts JSON body with 'query', optional 'count', and 'sentences'. Returns titles, publisher, final URL, publish date, and summary.",
      },
    ],
    rating: 4.5,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
  {
    id: "scholarai",
    owner_id: "scholarai-vendor",
    name: "ScholarAI Research & Literature Agent",
    description:
      "Searches scholarly corpora, returns metadata/full text, and generates annotated answers with citations.",
    capabilities: [
      "academic paper search",
      "scholarly literature review",
      "find research paper abstracts",
      "get full text of a paper",
      "ask questions to a PDF document",
      "find relevant patents",
      "analyze research projects",
      "scientific literature search",
    ],
    price_per_call_usd: 0.005,
    status: "active",
    endpoints: [
      {
        endpoint: "https://api.scholarai.io/api/abstracts",
        http_method: "GET",
        description:
          "Retrieves relevant abstracts and paper metadata by a search. Supports generative answers.",
      },
      {
        endpoint: "https://api.scholarai.io/api/fulltext",
        http_method: "GET",
        description:
          "Retrieves the full text of an article given its pdf_url.",
      },
      {
        endpoint: "https://api.scholarai.io/api/question",
        http_method: "GET",
        description: "Ask questions to a specific PDF document.",
      },
      {
        endpoint: "https://api.scholarai.io/api/patents",
        http_method: "GET",
        description: "Gets relevant patents using 2-6 keywords.",
      },
    ],
    rating: 4.5,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
  {
    id: "finance_agent",
    owner_id: "orbimesh-vendor",
    name: "Orbimesh Finance Agent",
    description:
      "Provides financial data, historical prices, dividends, market descriptions, and key financial metrics for stocks using Yahoo Finance.",
    capabilities: [
      "get historical stock prices",
      "get dividend and split history",
      "get market description",
      "get financial details",
    ],
    price_per_call_usd: 0.002,
    status: "active",
    endpoints: [
      {
        endpoint: "http://localhost:8010/history",
        http_method: "POST",
        description:
          "Returns historical OHLC prices for a given ticker, period, and interval.",
      },
      {
        endpoint: "http://localhost:8010/dividend",
        http_method: "POST",
        description:
          "Returns dividend and stock split history for a given ticker, period, and interval.",
      },
      {
        endpoint: "http://localhost:8010/market_desc",
        http_method: "GET",
        description:
          "Returns a long business summary for the given ticker.",
      },
      {
        endpoint: "http://localhost:8010/financial_details",
        http_method: "GET",
        description:
          "Returns key financial metrics for the given ticker.",
      },
    ],
    rating: 4.2,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
  {
    id: "wikipedia_agent",
    owner_id: "orbimesh-vendor",
    name: "Orbimesh Wikipedia Agent",
    description:
      "Searches and retrieves Wikipedia page titles, summaries, and URLs using the official Wikipedia API module.",
    capabilities: [
      "search Wikipedia pages",
      "get page summary by title",
      "get best-match summary (GET)",
      "get best-match summary (POST)",
    ],
    price_per_call_usd: 0.0015,
    status: "active",
    endpoints: [
      {
        endpoint: "http://localhost:8030/search/{query}",
        http_method: "GET",
        description:
          "Searches Wikipedia for a given query and returns a list of matching page titles.",
      },
      {
        endpoint: "http://localhost:8030/page/{title}",
        http_method: "GET",
        description:
          "Retrieves the summary of a specific Wikipedia page by its exact title.",
      },
      {
        endpoint: "http://localhost:8030/autosummary",
        http_method: "GET",
        description:
          "Fetches the best-matching Wikipedia page summary for a given query using query parameters.",
      },
      {
        endpoint: "http://localhost:8030/summary",
        http_method: "POST",
        description:
          "Fetches the best-matching Wikipedia page summary for a given query using a JSON request body.",
      },
    ],
    rating: 4.3,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
]
