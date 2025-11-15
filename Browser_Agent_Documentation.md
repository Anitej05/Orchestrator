# Browser Automation Agent Documentation

## Table of Contents
1. [Overview](#overview)
2. [Installation](#installation)
3. [Prerequisites](#prerequisites)
4. [Configuration](#configuration)
5. [Usage](#usage)
6. [Features](#features)
7. [API Endpoints](#api-endpoints)
8. [Advanced Features](#advanced-features)
9. [Troubleshooting](#troubleshooting)
10. [Best Practices](#best-practices)

## Overview

The Browser Automation Agent is a powerful custom browser automation tool that enables automated web interactions, data extraction, and web browsing tasks. It uses state-of-the-art (SOTA) techniques with multi-provider LLM fallback, vision capabilities, and live screenshot streaming.

### Key Features:
- Web browsing and navigation
- Element interaction (click, type, scroll, etc.)
- Data extraction from web pages
- Screenshot capture
- Live streaming of browser activity
- Multi-provider LLM support (Cerebras → Groq → NVIDIA)
- Vision capabilities (Ollama → NVIDIA)
- Task planning and execution

## Prerequisites

Before installing the Browser Automation Agent, ensure you have:

- Python 3.9 or higher
- pip package manager
- Git (for cloning the repository)
- At least 2GB of free disk space
- Stable internet connection

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/Orbimesh/Orbimesh-App.git
cd Orbimesh-App
```

### Step 2: Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### Step 3: Install Playwright Browser Drivers

The browser agent uses Playwright for browser automation. Install the required browser drivers:

```bash
playwright install chromium
```

This installs the Chromium browser which the agent will use for automation tasks.

### Step 4: Install Additional Dependencies (if needed)

If you encounter any missing dependencies, install them manually:

```bash
pip install playwright
playwright install-deps
```

## Configuration

### Environment Variables

Create a `.env` file in the backend directory with the following variables:

```env
# LLM API Keys (at least one required)
CEREBRAS_API_KEY=your_cerebras_api_key
GROQ_API_KEY=your_groq_api_key
NVIDIA_API_KEY=your_nvidia_api_key

# Vision Model API Keys (optional)
OLLAMA_API_KEY=your_ollama_api_key

# Port configuration
BROWSER_AGENT_PORT=8070
```

### Agent Configuration

The agent is configured in `backend/agents/browser_automation_agent.py`. You can modify the following settings:

- Port: The port on which the agent will run (default: 8070)
- Timeout settings: Adjust timeouts for various operations
- Headless mode: Set to True for headless browsing or False for visible browser

## Usage

### Starting the Browser Agent

To start the browser agent, run:

```bash
cd backend
python agents/browser_automation_agent.py
```

The agent will start on the configured port (default: 8070).

### Starting the Main Application

To start the main application with all agents:

```bash
cd backend
python main.py
```

This will start the main API server on port 8000 and automatically start all agents, including the browser agent.

### Using the Browser Agent

The browser agent can be used in several ways:

#### 1. Direct API Calls

Send a POST request to the agent endpoint:

```bash
curl -X POST http://localhost:8070/browse \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Go to google.com and search for Python tutorials",
    "max_steps": 10
  }'
```

#### 2. Through the Main Application

The browser agent is integrated into the main application and can be used through the orchestration system.

## API Endpoints

### `/browse` (POST)

Execute browser automation tasks.

**Request Body:**
```json
{
  "task": "string",
  "extract_data": "boolean",
  "max_steps": "integer"
}
```

**Parameters:**
- `task`: The task description for the browser agent (e.g., 'Go to leetcode.com/contest and find upcoming contests')
- `extract_data`: Whether to extract structured data from the page (default: False)
- `max_steps`: Maximum number of steps to execute (default: 10)

**Response:**
```json
{
  "success": "boolean",
  "task_summary": "string",
  "actions_taken": "array",
  "actions_planned": "array",
  "actions_succeeded": "array",
  "actions_failed": "array",
  "action_success_rate": "string",
  "screenshot_files": "array",
  "downloaded_files": "array",
  "uploaded_files": "array",
  "extracted_data": "object",
  "task_id": "string",
  "metrics": "object",
  "error": "string"
}
```

### `/health` (GET)

Check the health status of the agent.

### `/info` (GET)

Get information about the agent.

### `/live/{task_id}` (GET)

Get the latest live screenshot for a task.

## Features

### 1. Multi-Provider LLM Fallback

The agent uses a chain of LLM providers with automatic fallback:
- Primary: Cerebras
- Fallback 1: Groq
- Fallback 2: NVIDIA

If one provider fails, the agent automatically tries the next provider in the chain.

### 2. Vision Capabilities

The agent supports vision-based analysis with the following providers:
- Primary: Ollama
- Fallback: NVIDIA

Vision capabilities are used for:
- Image analysis
- Complex UI element detection
- CAPTCHA solving
- Visual content understanding

### 3. Task Planning

The agent creates an initial plan by breaking down tasks into subtasks:
- Navigates to required websites
- Performs specific actions
- Extracts required data
- Completes complex workflows

### 4. Live Streaming

The agent supports live streaming of browser activity to the canvas with:
- Real-time screenshots
- Task progress tracking
- Action visualization

### 5. Advanced Element Selection

The agent uses multiple strategies to select elements:
- CSS selectors
- XPath
- Text-based selection
- JavaScript execution
- Coordinate-based clicks

### 6. Page Stabilization

The agent waits for pages to stabilize before performing actions:
- Waits for network idle
- Handles dynamic content loading
- Dismisses overlays and modals

## Advanced Features

### 1. Context Optimization

The agent uses a ContextOptimizer to reduce token usage by 70% by filtering relevant elements and focusing on the current task.

### 2. Dynamic Planning

The agent can replan tasks dynamically when:
- Stuck on the same action
- High failure rate
- Current subtask is impossible

### 3. Vision Optimization

The agent uses smart vision usage to reduce costs by 80% with rule-based vision decisions.

### 4. SOTA Browser Automation

The agent includes several state-of-the-art improvements:
- Context optimization
- Multi-strategy selectors
- Page stabilization
- Dynamic planning
- Vision optimization

## Troubleshooting

### Common Issues

#### 1. Browser Agent Won't Start

**Symptoms:**
- Agent fails to start
- Port already in use error

**Solutions:**
1. Check if another instance is running:
   ```bash
   netstat -an | grep 8070
   ```
2. Kill any existing processes on the port
3. Ensure Playwright is properly installed:
   ```bash
   playwright install chromium
   ```

#### 2. Element Selection Failures

**Symptoms:**
- Agent fails to click or interact with elements
- "Element not found" errors

**Solutions:**
1. Verify the element exists and is visible
2. Check for dynamic content loading
3. Use more specific selectors
4. Ensure proper page stabilization

#### 3. API Key Issues

**Symptoms:**
- LLM calls fail
- "Invalid API key" errors

**Solutions:**
1. Verify API keys are correctly set in `.env`
2. Check API key validity with the provider
3. Ensure sufficient credits/balance

#### 4. Vision Model Issues

**Symptoms:**
- Vision analysis fails
- High latency for visual tasks

**Solutions:**
1. Verify vision API keys are set
2. Check vision model availability
3. Consider using text-only mode for non-visual tasks

### Debugging Tips

1. **Check Logs:** Examine the agent logs in the `logs` directory
2. **Enable Verbose Mode:** Set logging level to DEBUG in the agent code
3. **Test Incrementally:** Start with simple tasks before complex workflows
4. **Monitor Resources:** Ensure sufficient memory and CPU for browser automation

## Best Practices

### 1. Task Design

- Be specific with task descriptions
- Break complex tasks into smaller steps
- Include expected outcomes in the task description
- Specify required data to extract

### 2. Error Handling

- Implement proper error handling in your application
- Monitor action success rates
- Plan for fallback scenarios
- Log important events for debugging

### 3. Performance Optimization

- Use appropriate timeout values
- Limit max_steps for simple tasks
- Consider headless mode for production
- Monitor resource usage

### 4. Security

- Store API keys securely
- Validate input parameters
- Sanitize extracted data
- Implement rate limiting

### 5. Testing

- Test with various websites and scenarios
- Verify data extraction accuracy
- Check for edge cases
- Monitor for unexpected behaviors

## Example Usage

### Example 1: Simple Navigation

```json
{
  "task": "Go to https://www.python.org and take a screenshot",
  "max_steps": 5
}
```

### Example 2: Data Extraction

```json
{
  "task": "Go to https://news.ycombinator.com and extract the top 5 headlines",
  "extract_data": true,
  "max_steps": 10
}
```

### Example 3: Form Interaction

```json
{
  "task": "Go to https://httpbin.org/forms/post and fill out the form with name 'John' and email 'john@example.com'",
  "max_steps": 15
}
```

## Conclusion

The Browser Automation Agent is a powerful tool for automating web interactions and data extraction. With its advanced features like multi-provider LLM support, vision capabilities, and task planning, it can handle complex web automation tasks efficiently.
