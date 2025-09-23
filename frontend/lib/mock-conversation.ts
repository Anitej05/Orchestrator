// Mock data for interactive conversation demo

export interface MockConversationStep {
  id: string;
  userInput: string;
  systemQuestion?: string;
  finalResponse?: string;
  taskAgentPairs?: any[];
  requiresInput: boolean;
}

export const mockConversationFlows: Record<string, MockConversationStep[]> = {
  "sales_analysis": [
    {
      id: "step1",
      userInput: "Help me analyze sales data",
      systemQuestion: "What type of sales data would you like to analyze? (e.g., monthly revenue, product performance, customer segments)",
      requiresInput: true
    },
    {
      id: "step2", 
      userInput: "monthly revenue",
      systemQuestion: "What time period should I analyze? (e.g., last 6 months, year-to-date, custom range)",
      requiresInput: true
    },
    {
      id: "step3",
      userInput: "last 6 months",
      finalResponse: "I'll help you analyze your monthly revenue for the last 6 months. Here's your workflow:",
      taskAgentPairs: [
        {
          task_name: "data_extraction",
          primary: {
            id: "data-agent-1",
            name: "Sales Data Extractor", 
            price_per_call_usd: 0.002,
            capabilities: ["Data Analysis", "Revenue Analysis"]
          }
        },
        {
          task_name: "trend_analysis", 
          primary: {
            id: "analytics-agent-1",
            name: "Trend Analyzer",
            price_per_call_usd: 0.003,
            capabilities: ["Statistical Analysis", "Trend Detection"]
          }
        },
        {
          task_name: "report_generation",
          primary: {
            id: "report-agent-1", 
            name: "Report Generator",
            price_per_call_usd: 0.002,
            capabilities: ["Report Generation", "Data Visualization"]
          }
        }
      ],
      requiresInput: false
    }
  ],
  
  "marketing_campaign": [
    {
      id: "step1",
      userInput: "Create a marketing campaign",
      systemQuestion: "What type of product or service are you marketing? Please provide details about your target audience.",
      requiresInput: true
    },
    {
      id: "step2",
      userInput: "SaaS productivity tool for small businesses",
      systemQuestion: "What's your campaign budget and preferred channels? (e.g., social media, email, content marketing)",
      requiresInput: true  
    },
    {
      id: "step3",
      userInput: "5000 budget, focus on social media and email",
      finalResponse: "Perfect! I'll create a comprehensive marketing campaign for your SaaS productivity tool. Here's your workflow:",
      taskAgentPairs: [
        {
          task_name: "audience_research",
          primary: {
            id: "research-agent-1",
            name: "Market Research Agent",
            price_per_call_usd: 0.004,
            capabilities: ["Market Research", "Audience Analysis"]
          }
        },
        {
          task_name: "content_creation",
          primary: {
            id: "content-agent-1",
            name: "Content Creator",
            price_per_call_usd: 0.003,
            capabilities: ["Content Creation", "Social Media"]
          }
        },
        {
          task_name: "email_campaign",
          primary: {
            id: "email-agent-1",
            name: "Email Marketing Specialist", 
            price_per_call_usd: 0.002,
            capabilities: ["Email Marketing", "Automation"]
          }
        }
      ],
      requiresInput: false
    }
  ],

  "default": [
    {
      id: "step1",
      userInput: "",
      systemQuestion: "I'd like to understand your needs better. Could you provide more details about what you're trying to accomplish?",
      requiresInput: true
    },
    {
      id: "step2",
      userInput: "",
      finalResponse: "Based on your requirements, here's a suggested workflow:",
      taskAgentPairs: [
        {
          task_name: "general_task",
          primary: {
            id: "general-agent-1",
            name: "General Purpose Agent",
            price_per_call_usd: 0.002,
            capabilities: ["General Processing", "Task Execution"]
          }
        }
      ],
      requiresInput: false
    }
  ]
};

export function findMatchingFlow(userInput: string): string {
  const input = userInput.toLowerCase();
  
  if (input.includes("sales") || input.includes("revenue") || input.includes("data")) {
    return "sales_analysis";
  }
  
  if (input.includes("marketing") || input.includes("campaign") || input.includes("promote")) {
    return "marketing_campaign";
  }
  
  return "default";
}

export function getNextStep(flowId: string, currentStep: number, userInput: string): MockConversationStep | null {
  const flow = mockConversationFlows[flowId];
  if (!flow || currentStep >= flow.length) return null;
  
  const step = { ...flow[currentStep] };
  if (currentStep > 0) {
    step.userInput = userInput;
  }
  
  return step;
}
