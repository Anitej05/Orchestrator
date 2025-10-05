"use client"

import { useState, useEffect, useCallback, useRef } from "react"
import { Database, Users, BarChart3, Lightbulb, Loader2 } from "lucide-react"
import type { TaskAgentPair } from "@/lib/types"

interface ExecutionResult {
  taskId: string
  taskDescription: string
  agentName: string
  status: string
  output: string
  cost: number
  executionTime: number
}

interface WorkflowOrchestrationProps {
  isExecuting: boolean
  isDryRun: boolean
  taskAgentPairs: TaskAgentPair[]
  selectedAgents: Record<string, string>
  taskInput?: string
  onComplete: (results: ExecutionResult[]) => void
  onThreadIdUpdate?: (threadId: string) => void
  onExecutionResultsUpdate?: (results: ExecutionResult[]) => void
}

interface DataSource {
  name: string
  icon: string
  description: string
  status: "loading" | "connected" | "pending"
  color: string
}

interface TaskSegment {
  name: string
  agent: string
  estimatedCost: number
}

interface TaskInsight {
  text: string
}

export default function WorkflowOrchestration({
  isExecuting,
  isDryRun,
  taskAgentPairs,
  selectedAgents,
  taskInput = "",
  onComplete,
  onThreadIdUpdate,
  onExecutionResultsUpdate,
}: WorkflowOrchestrationProps) {
  // WebSocket ref for connection management (must be inside component)
  const wsRef = useRef<WebSocket | null>(null)
  const [currentPhase, setCurrentPhase] = useState<"" | "intro" | "loading" | "analysis" | "insights">("intro")
  const [dataSources, setDataSources] = useState<DataSource[]>([])
  const [taskAnalysis, setTaskAnalysis] = useState("")
  const [taskSegments, setTaskSegments] = useState<TaskSegment[]>([])
  const [taskInsights, setTaskInsights] = useState<TaskInsight[]>([])
  const [displayText, setDisplayText] = useState("")
  const [elapsedTime, setElapsedTime] = useState(0)
  const [executionStarted, setExecutionStarted] = useState(false)
  const [finalMarkdown, setFinalMarkdown] = useState<string | null>(null)

  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const executionRef = useRef<boolean>(false)

  // Timer for elapsed time
  useEffect(() => {
    if (isExecuting) {
      timerRef.current = setInterval(() => {
        setElapsedTime((prev) => prev + 0.1)
      }, 100)
    } else {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }

    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }, [isExecuting])

  // Reset state when execution starts
  useEffect(() => {
    if (isExecuting && !executionStarted) {
      setCurrentPhase("intro")
      setDataSources([])
      setTaskAnalysis("")
      setTaskSegments([])
      setTaskInsights([])
      setDisplayText("")
      setElapsedTime(0)
      setExecutionStarted(true)
      executionRef.current = false
    } else if (!isExecuting) {
      setExecutionStarted(false)
    }
  }, [isExecuting, executionStarted])

  // Start execution once when conditions are met
  useEffect(() => {
    if (isExecuting && executionStarted && !executionRef.current && taskAgentPairs.length > 0) {
      executionRef.current = true
      handleSubmit()
    }
    // Cleanup WebSocket on unmount
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
        wsRef.current = null
      }
    }
  }, [isExecuting, executionStarted, taskAgentPairs.length])

  // Initialize task segments and insights when not executing
  useEffect(() => {
    if (!isExecuting && taskAgentPairs.length > 0) {
      // Ensure all data is populated for display after execution
      if (taskSegments.length === 0) {
        setTaskSegments(generateTaskSegments())
      }
      if (taskInsights.length === 0) {
        setTaskInsights(generateTaskInsights())
      }
      if (!taskAnalysis) {
        setTaskAnalysis(generateTaskAnalysis())
      }
      if (dataSources.length === 0) {
        setDataSources(generateTaskDataSources())
      }
    }
  }, [isExecuting, taskAgentPairs.length])

  const typeText = useCallback(async (text: string, delay = 30) => {
    return new Promise<void>((resolve) => {
      let currentIndex = 0
      const typeInterval = setInterval(() => {
        if (currentIndex <= text.length) {
          setDisplayText(text.slice(0, currentIndex))
          currentIndex++
        } else {
          clearInterval(typeInterval)
          resolve()
        }
      }, delay)
    })
  }, [])

  const getSelectedAgent = (taskName: string) => {
    const selectedAgentId = selectedAgents[taskName]
    const pair = taskAgentPairs.find((p) => p.task_name === taskName)
    if (!pair) return null

    if (pair.primary.id === selectedAgentId) return pair.primary
    return pair.fallbacks.find((agent) => agent.id === selectedAgentId) || pair.primary
  }

  const generateTaskDataSources = () => {
    const agentTypes = new Set<string>()
    const capabilities = new Set<string>()

    taskAgentPairs.forEach((pair) => {
      const agent = getSelectedAgent(pair.task_name)
      if (agent) {
        agentTypes.add(agent.name)
        agent.capabilities.forEach((cap) => capabilities.add(cap))
      }
    })

    const sources: DataSource[] = []

    // Add agent registry as primary source
    sources.push({
      name: "Agent Registry",
      icon: "🤖",
      description: `${taskAgentPairs.length} tasks identified for execution`,
      status: "connected",
      color: "text-green-500",
    })

    // Add capability-based sources
    if (capabilities.has("find_travel_agent") || capabilities.has("research_sales_leads")) {
      sources.push({
        name: "Search Engine",
        icon: "🔍",
        description: "Web search and data aggregation capabilities",
        status: "connected",
        color: "text-green-500",
      })
    }

    if (capabilities.has("draft_marketing_emails") || capabilities.has("Email Drafting")) {
      sources.push({
        name: "Email Templates",
        icon: "📧",
        description: "Professional email templates and formatting",
        status: "connected",
        color: "text-green-500",
      })
    }

    if (capabilities.has("Translation") || capabilities.has("Content Creation")) {
      sources.push({
        name: "Language Models",
        icon: "🌐",
        description: "Multi-language processing and content generation",
        status: "connected",
        color: "text-green-500",
      })
    }

    if (capabilities.has("Data Analysis") || capabilities.has("Research")) {
      sources.push({
        name: "Analytics Engine",
        icon: "📊",
        description: "Data processing and analysis capabilities",
        status: "connected",
        color: "text-green-500",
      })
    }

    // Fallback if no specific capabilities matched
    if (sources.length === 1) {
      sources.push({
        name: "Task Processor",
        icon: "⚙️",
        description: "General task execution framework",
        status: "connected",
        color: "text-green-500",
      })
    }

    return sources
  }

  const generateTaskAnalysis = () => {
    if (taskAgentPairs.length === 0) {
      return "No tasks available for analysis."
    }

    const totalCost = taskAgentPairs.reduce((sum, pair) => {
      const agent = getSelectedAgent(pair.task_name)
      return sum + (agent?.price_per_call_usd || 0)
    }, 0)

    const agentNames = taskAgentPairs.map((pair) => {
      const agent = getSelectedAgent(pair.task_name)
      return agent?.name || "Unknown Agent"
    })

    const uniqueAgents = [...new Set(agentNames)]
    const taskNames = taskAgentPairs.map((pair) =>
      pair.task_name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
    )

    const avgRating =
      taskAgentPairs.reduce((sum, pair) => {
        const agent = getSelectedAgent(pair.task_name)
        return sum + (agent?.rating || 0)
      }, 0) / taskAgentPairs.length

    return `Processing ${taskAgentPairs.length} task${taskAgentPairs.length !== 1 ? "s" : ""} across ${uniqueAgents.length} specialized agent${uniqueAgents.length !== 1 ? "s" : ""}. Tasks include: ${taskNames.join(", ")}. Total estimated cost: $${totalCost.toFixed(2)}. Average agent rating: ${avgRating.toFixed(1)}/5.0. Execution mode: ${isDryRun ? "Simulation" : "Live"}. All agents are currently active and ready for task execution.`
  }

  const generateTaskSegments = () => {
    return taskAgentPairs.map((pair) => {
      const agent = getSelectedAgent(pair.task_name)
      return {
        name: pair.task_name.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
        agent: agent?.name || "Unknown Agent",
        estimatedCost: agent?.price_per_call_usd || 0,
      }
    })
  }

  const generateTaskInsights = () => {
    const insights: TaskInsight[] = []

    if (taskAgentPairs.length === 0) {
      return [{ text: "No tasks available for insight generation." }]
    }

    // Cost analysis
    const costs = taskAgentPairs.map((pair) => {
      const agent = getSelectedAgent(pair.task_name)
      return agent?.price_per_call_usd || 0
    })
    const totalCost = costs.reduce((sum, cost) => sum + cost, 0)
    const avgCost = totalCost / costs.length
    const maxCost = Math.max(...costs)
    const minCost = Math.min(...costs)

    if (maxCost > avgCost * 1.5) {
      const expensiveTask = taskAgentPairs.find((pair) => {
        const agent = getSelectedAgent(pair.task_name)
        return (agent?.price_per_call_usd || 0) === maxCost
      })
      if (expensiveTask) {
        insights.push({
          text: `Cost optimization opportunity - ${expensiveTask.task_name.replace(/_/g, " ")} represents ${((maxCost / totalCost) * 100).toFixed(0)}% of total workflow cost at $${maxCost}. Consider reviewing agent selection or task complexity.`,
        })
      }
    }

    // Agent diversity analysis
    const agentIds = new Set(
      taskAgentPairs.map((pair) => {
        const agent = getSelectedAgent(pair.task_name)
        return agent?.id || "unknown"
      }),
    )

    if (agentIds.size < taskAgentPairs.length) {
      insights.push({
        text: `Agent efficiency detected - ${agentIds.size} agent${agentIds.size !== 1 ? "s" : ""} handling ${taskAgentPairs.length} tasks. This consolidation may reduce context switching overhead and improve execution speed.`,
      })
    }

    // Capability analysis
    const allCapabilities = new Set<string>()
    taskAgentPairs.forEach((pair) => {
      const agent = getSelectedAgent(pair.task_name)
      if (agent) {
        agent.capabilities.forEach((cap) => allCapabilities.add(cap))
      }
    })

    if (allCapabilities.size > taskAgentPairs.length) {
      insights.push({
        text: `Rich capability coverage - ${allCapabilities.size} unique capabilities across ${taskAgentPairs.length} tasks. This suggests comprehensive agent selection with potential for cross-task optimization.`,
      })
    }

    // Execution mode insight
    insights.push({
      text: `Execution strategy - ${isDryRun ? "Dry run mode" : "Live execution mode"} selected. ${isDryRun ? "This will simulate all operations without making actual API calls, allowing safe testing of the workflow." : "This will execute all tasks with real API calls and generate actual results."}`,
    })

    return insights.length > 0
      ? insights
      : [{ text: "Workflow analysis complete. All tasks are properly configured and ready for execution." }]
  }

  // WebSocket message handler
  const handleStreamEvent = useCallback(async (event: any) => {
    // event structure: { node, data, thread_id, status, timestamp, ... }
    const node = event.node
    const nodeData = event.data || {}
    const threadId = event.thread_id || nodeData.thread_id
    const progress = nodeData.progress_percentage || null
    const nodeSeq = nodeData.node_sequence || null
    console.log("Received WebSocket event:", event)

    // Update thread ID in parent component if available
    if (threadId && onThreadIdUpdate) {
      onThreadIdUpdate(threadId)
    }

    switch (node) {
      case "__start__":
        setCurrentPhase("intro")
        await typeText(event.message || nodeData.description || "Starting agent orchestration...")
        break
      case "parse_prompt":
        setCurrentPhase("loading")
        const sources = generateTaskDataSources()
        setDataSources(sources.map(s => ({ ...s, status: "loading" as const })))
        await typeText(nodeData.description || "Parsing your request and identifying tasks...")
        break
      case "agent_directory_search":
        setDataSources(prev => prev.map(s => ({ ...s, status: "connected" as const })))
        await typeText(nodeData.description || "Searching for suitable agents in the directory...")
        break
      case "rank_agents":
        setCurrentPhase("analysis")
        const analysis = generateTaskAnalysis()
        await typeText(nodeData.description || analysis, 20)
        setTaskAnalysis(analysis)
        setTaskSegments(generateTaskSegments())
        break
      case "plan_execution":
        setCurrentPhase("insights")
        setTaskInsights(generateTaskInsights())
        await typeText(nodeData.description || "Generating execution plan and insights...")
        break
      case "__end__":
        // Use nodeData for final results and summary
        const pairs = event.task_agent_pairs || nodeData.task_agent_pairs || []
        const finalResponse = event.final_response || nodeData.final_response || ""
        const summary = event.summary || nodeData.summary || {}
        setFinalMarkdown(finalResponse);

        // Generate execution results from the task agent pairs
        if (pairs.length > 0) {
          const results: ExecutionResult[] = pairs.map((pair: any, index: number) => ({
            taskId: pair.task_name || `task_${index}`,
            taskDescription: pair.task_name?.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()) || "Unknown Task",
            agentName: pair.primary?.name || pair.agent?.name || "Unknown Agent",
            status: isDryRun ? "dry-run-success" : "success",
            output: isDryRun
              ? `[DRY RUN] Would execute: ${pair.task_name?.replace(/_/g, " ")} using ${pair.primary?.name || "Unknown Agent"}`
              : finalResponse || `Successfully completed: ${pair.task_name?.replace(/_/g, " ")}`,
            cost: pair.primary?.price_per_call_usd || 0,
            executionTime: Math.floor((Math.random() * 5 + 3) * 10) / 10,
          }))

          // Update execution results in parent component
          if (onExecutionResultsUpdate) {
            onExecutionResultsUpdate(results)
          }

          // Optionally, you can display summary stats here
          if (summary && summary.execution_completed) {
            setTaskAnalysis(`Workflow completed. Total tasks: ${summary.total_tasks}, Estimated cost: $${summary.total_estimated_cost}, Agents: ${summary.agent_names?.join(", ")}`)
          }
          onComplete(results)
        } else {
          // Fallback to generating results from taskAgentPairs if no pairs in response
          const results: ExecutionResult[] = taskAgentPairs.map((pair) => {
            const selectedAgent = getSelectedAgent(pair.task_name)
            return {
              taskId: pair.task_name,
              taskDescription: pair.task_name.replace(/_/g, " ").replace(/\b\w/g, (l: string) => l.toUpperCase()),
              agentName: selectedAgent?.name || "Unknown Agent",
              status: isDryRun ? "dry-run-success" : "success",
              output: isDryRun
                ? `[DRY RUN] Would execute: ${pair.task_name.replace(/_/g, " ")} using ${selectedAgent?.name || "Unknown Agent"}`
                : finalResponse || `Successfully completed: ${pair.task_name.replace(/_/g, " ")}`,
              cost: selectedAgent?.price_per_call_usd || 0,
              executionTime: Math.floor((Math.random() * 5 + 3) * 10) / 10,
            }
          })

          // Update execution results in parent component
          if (onExecutionResultsUpdate) {
            onExecutionResultsUpdate(results)
          }
          onComplete(results)
        }
        break
      case "__error__":
        // Display backend error to user
        const errorMsg = event.message || nodeData.description || event.error || "An unknown error occurred."
        setCurrentPhase("intro")
        setDisplayText(errorMsg)
        console.error("WebSocket error:", errorMsg)
        break
      default:
        // For other nodes, show progress and description
        await typeText(nodeData.description || `Processing ${node.replace(/_/g, " ")}...`)
    }
  }, [isDryRun, taskAgentPairs, onComplete, onThreadIdUpdate, onExecutionResultsUpdate])

  // Refactored orchestration logic using WebSocket
  const handleSubmit = useCallback(() => {
    // Only connect if not already connected
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    const ws = new window.WebSocket("ws://localhost:8000/ws/chat")
    wsRef.current = ws

    ws.onopen = () => {
      console.log("WebSocket connected, sending initial prompt...")
      // Send initial prompt when connection opens
      const prompt = taskInput || "Please analyze and execute the requested workflow"
      ws.send(JSON.stringify({ prompt }))
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        // Pass parsed data to the existing stream handler
        handleStreamEvent(data)
      } catch (err) {
        console.error("WebSocket message parse error:", err)
      }
    }

    ws.onclose = (event) => {
      console.log("WebSocket closed:", event)
      wsRef.current = null
    }

    ws.onerror = (event) => {
      console.error("WebSocket error:", event)
    }
  }, [taskInput, handleStreamEvent])

  return (
    <div className="space-y-6">
      {/* Introduction Phase */}
      {(currentPhase === "intro" || (!isExecuting && currentPhase !== "")) && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
              <span className="text-lg">🤖</span>
            </div>
            <div className="flex-1">
              <p className="text-gray-700 leading-relaxed">
                {displayText ||
                  (taskAgentPairs.length > 0
                    ? `I executed your ${taskAgentPairs.length}-step workflow: ${taskAgentPairs.map((pair) => pair.task_name.replace(/_/g, " ")).join(", ")}. Connected to required data sources and agents.`
                    : "Workflow execution completed.")}
                {isExecuting && <span className="animate-pulse">|</span>}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Loading Phase */}
      {currentPhase === "loading" && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Database className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-medium text-gray-900">Loading data sources...</h3>
          </div>
          <div className="space-y-3">
            {dataSources.map((source, index) => (
              <div key={index} className="flex items-center space-x-3">
                <div
                  className={`w-2 h-2 rounded-full ${source.status === "connected" ? "bg-green-500" : "bg-yellow-500"}`}
                />
                <span className="text-2xl">{source.icon}</span>
                <div className="flex-1">
                  <span className="font-medium text-gray-900">{source.name}</span>
                  <span className="text-gray-500 ml-2">
                    {source.status === "loading" ? (
                      <span className="flex items-center">
                        <Loader2 className="w-3 h-3 animate-spin mr-1" />
                        Loading...
                      </span>
                    ) : (
                      source.description
                    )}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Data Sources Connected */}
      {currentPhase !== "intro" && dataSources.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Database className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-medium text-gray-900">Data sources connected</h3>
          </div>
          <div className="space-y-3">
            {dataSources.map((source, index) => (
              <div key={index} className="flex items-center space-x-3">
                <div className="w-2 h-2 rounded-full bg-green-500" />
                <span className="text-2xl">{source.icon}</span>
                <div className="flex-1">
                  <span className="font-medium text-gray-900">{source.name}</span>
                  <span className="text-gray-500 ml-2">{source.description}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Task Analysis */}
      {(currentPhase === "analysis" || currentPhase === "insights" || (!isExecuting && taskAnalysis)) && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Users className="w-5 h-5 text-green-500" />
            <h3 className="text-lg font-medium text-gray-900">Task Analysis</h3>
            <span className="text-green-600 font-semibold">{taskAgentPairs.length}</span>
          </div>
          <p className="text-gray-700 leading-relaxed">
            {currentPhase === "analysis" && isExecuting ? (
              <>
                {displayText}
                <span className="animate-pulse">|</span>
              </>
            ) : (
              taskAnalysis || generateTaskAnalysis()
            )}
          </p>
        </div>
      )}

      {/* Task Segments */}
      {taskSegments.length > 0 && (
        <div className="bg-white rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <BarChart3 className="w-5 h-5 text-blue-500" />
            <h3 className="text-lg font-medium text-gray-900">Task Segments</h3>
          </div>
          <div className="space-y-2">
            {taskSegments.map((segment, index) => (
              <div key={index} className="flex items-center space-x-2">
                <div className="w-2 h-2 rounded-full bg-blue-500 mt-2 flex-shrink-0" />
                <span className="text-gray-700">
                  {segment.name}{" "}
                  <span className="text-gray-500">
                    ({segment.agent} - ${segment.estimatedCost})
                  </span>
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Key Insights */}
      {(currentPhase === "insights" || (!isExecuting && taskInsights.length > 0)) && (
        <div className="bg-gray-50 rounded-lg border border-gray-200 p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Lightbulb className="w-5 h-5 text-purple-500" />
            <h3 className="text-lg font-medium text-gray-900">Key Insights</h3>
          </div>
          <div className="space-y-4">
            {(taskInsights.length > 0 ? taskInsights : generateTaskInsights()).map((insight, index) => (
              <div key={index} className="flex items-start space-x-3">
                <div className="w-2 h-2 rounded-full bg-purple-500 mt-2 flex-shrink-0" />
                <p className="text-gray-700 leading-relaxed">{insight.text}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Status Bar */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div
                className={`w-2 h-2 rounded-full ${isExecuting ? "bg-green-400 animate-pulse" : "bg-green-500"}`}
              ></div>
              <span className={`font-medium ${isExecuting ? "text-green-600" : "text-green-700"}`}>
                {isExecuting
                  ? isDryRun
                    ? "DRY RUN MODE"
                    : "LIVE EXECUTION"
                  : isDryRun
                    ? "DRY RUN COMPLETED"
                    : "EXECUTION COMPLETED"}
              </span>
            </div>
            <div className="text-gray-500">
              {isExecuting ? (
                <>
                  {currentPhase === "intro" && "Initializing workflow..."}
                  {currentPhase === "loading" && "Connecting to agents..."}
                  {currentPhase === "analysis" && "Analyzing tasks..."}
                  {currentPhase === "insights" && "Generating insights..."}
                </>
              ) : (
                "Workflow execution finished"
              )}
            </div>
          </div>
          <div className="text-gray-500">
            {elapsedTime.toFixed(1)}s {isExecuting ? "elapsed" : "total"}
          </div>
        </div>
      </div>

      {/* Final Response */}
      {/* {finalMarkdown && (
        <div className="bg-white rounded-lg border border-gray-200 p-6 mt-4">
          <h3 className="text-lg font-medium text-gray-900 mb-4">Final Response</h3>
          <div className="prose max-w-none">
            <ReactMarkdown
              components={{
                a: ({ href, children }) => (
                  <a href={href} target="_blank" rel="noopener noreferrer" className="text-blue-600 underline mx-2">{children}</a>
                ),
                table: ({ children }) => (
                  <table className="min-w-full border border-gray-300 rounded overflow-hidden text-sm my-6">{children}</table>
                ),
                thead: ({ children }) => (
                  <thead className="bg-gray-100 text-gray-700 font-semibold">{children}</thead>
                ),
                tbody: ({ children }) => <tbody>{children}</tbody>,
                tr: ({ children }) => <tr className="border-b last:border-b-0">{children}</tr>,
                th: ({ children }) => <th className="px-4 py-3 text-left border-b border-gray-200">{children}</th>,
                td: ({ children }) => <td className="px-4 py-3 border-b border-gray-100">{children}</td>,
                p: ({ children }) => <p className="mb-6 leading-relaxed">{children}</p>,
                ul: ({ children }) => <ul className="list-disc ml-8 mb-4">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal ml-8 mb-4">{children}</ol>,
                li: ({ children }) => <li className="mb-2">{children}</li>,
                h1: ({ children }) => <h1 className="text-2xl font-bold text-gray-900 mt-8 mb-4">{children}</h1>,
                h2: ({ children }) => <h2 className="text-xl font-semibold text-gray-900 mt-6 mb-3">{children}</h2>,
                h3: ({ children }) => <h3 className="text-lg font-medium text-gray-900 mt-4 mb-2">{children}</h3>,
                hr: () => <hr className="my-8 border-gray-300" />,
              }}
            >
              {finalMarkdown}
            </ReactMarkdown>
          </div>
        </div> 
      )}*/}
    </div>
  )
}
