import type { Agent } from "./types"

// Static data constants for the application
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
export const fallbackAgents: Agent[] = [
  {
    id: "lead-hunter-pro",
    owner_id: "demo_owner",
    name: "Lead Hunter Pro",
    description:
      "Advanced lead generation agent that finds high-quality prospects using multiple data sources and AI-powered qualification.",
    capabilities: ["research_sales_leads", "Lead Generation", "Data Analysis"],
    price_per_call_usd: 2.5,
    status: "active",
    endpoints: [
      {
        endpoint: "https://api.leadhunter.example.com/search",
        http_method: "POST",
        description: "Main lead search endpoint",
      },
    ],
    rating: 4.8,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
  {
    id: "email-craft-ai",
    owner_id: "demo_owner",
    name: "EmailCraft AI",
    description:
      "Professional email drafting agent that creates personalized, engaging emails for sales, marketing, and customer communication.",
    capabilities: ["draft_marketing_emails", "Email Drafting", "Content Creation"],
    price_per_call_usd: 1.0,
    status: "active",
    endpoints: [
      {
        endpoint: "https://api.emailcraft.example.com/draft",
        http_method: "POST",
        description: "Email drafting endpoint",
      },
    ],
    rating: 4.9,
    public_key_pem:
      "-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA...\n-----END PUBLIC KEY-----",
  },
]

// Legacy export for backward compatibility
export const agents = fallbackAgents;
