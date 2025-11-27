"use client"

import { useState, useEffect } from "react"
import { useUser } from "@clerk/nextjs"
import AppSidebar from "@/components/app-sidebar"
import Navbar from "@/components/navbar"
import { SidebarProvider, SidebarInset, SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { 
  Activity, 
  MessageSquare, 
  Workflow, 
  Users, 
  TrendingUp, 
  Clock,
  CheckCircle,
  AlertCircle,
  BarChart3,
  DollarSign,
  Target,
  Zap
} from "lucide-react"
import { 
  BarChart, 
  Bar, 
  LineChart, 
  Line, 
  PieChart, 
  Pie, 
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer 
} from "recharts"

interface DashboardMetrics {
  totalConversations: number
  totalWorkflows: number
  totalAgents: number
  recentActivity: number
  conversationTrend: Array<{ date: string; count: number }>
  workflowStatus: Array<{ name: string; value: number }>
  agentUsage: Array<{ name: string; calls: number }>
  recentConversations: Array<{ id: string; title: string; date: string; status: string }>
  costMetrics: {
    today: number
    week: number
    month: number
    total: number
    avgPerConversation: number
  }
  costTrend: Array<{ date: string; cost: number }>
  performanceMetrics: {
    totalTasks: number
    successfulTasks: number
    failedTasks: number
    successRate: number
    avgResponseTime: number
    avgTasksPerConversation: number
  }
  topAgents: Array<{ name: string; calls: number; cost: number; costPerCall: number }>
  hourlyUsage: Array<{ hour: string; count: number }>
}

const COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

function MetricsContent() {
  const { user } = useUser()
  const { open } = useSidebar()
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null)
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState("overview")

  useEffect(() => {
    const fetchMetrics = async () => {
      if (!user) return
      
      setLoading(true)
      try {
        const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'
        
        // Fetch dashboard metrics from the new endpoint
        const metricsRes = await fetch(`${API_URL}/api/metrics/dashboard`, {
          method: 'GET',
          headers: {
            'X-User-ID': user.id,
            'Content-Type': 'application/json'
          }
        })
        
        if (metricsRes.ok) {
          const data = await metricsRes.json()
          
          // Fetch agents for usage data
          const agentsRes = await fetch(`${API_URL}/api/agents/all`, {
            method: 'GET',
            headers: {
              'Content-Type': 'application/json'
            }
          })
          const agents = agentsRes.ok ? await agentsRes.json() : []
          
          // Process agent usage (mock data - would need execution logs in real implementation)
          const agentUsage = agents.slice(0, 5).map((agent: any) => ({
            name: agent.name.length > 15 ? agent.name.substring(0, 15) + '...' : agent.name,
            calls: Math.floor(Math.random() * 100) // Mock data
          }))
          
          setMetrics({
            totalConversations: data.total_conversations || 0,
            totalWorkflows: data.total_workflows || 0,
            totalAgents: data.total_agents || 0,
            recentActivity: data.recent_activity || 0,
            conversationTrend: data.conversation_trend || [],
            workflowStatus: data.workflow_status || [],
            agentUsage: data.top_agents || [],
            recentConversations: data.recent_conversations || [],
            costMetrics: data.cost_metrics || {
              today: 0,
              week: 0,
              month: 0,
              total: 0,
              avgPerConversation: 0
            },
            costTrend: data.cost_trend || [],
            performanceMetrics: data.performance_metrics || {
              totalTasks: 0,
              successfulTasks: 0,
              failedTasks: 0,
              successRate: 0,
              avgResponseTime: 0,
              avgTasksPerConversation: 0
            },
            topAgents: data.top_agents || [],
            hourlyUsage: data.hourly_usage || []
          })
        } else {
          const errorText = await metricsRes.text()
          console.error('Failed to fetch metrics:', metricsRes.status, errorText)
          // Set empty metrics on error
          setMetrics({
            totalConversations: 0,
            totalWorkflows: 0,
            totalAgents: 0,
            recentActivity: 0,
            conversationTrend: [],
            workflowStatus: [],
            agentUsage: [],
            recentConversations: [],
            costMetrics: { today: 0, week: 0, month: 0, total: 0, avgPerConversation: 0 },
            costTrend: [],
            performanceMetrics: { totalTasks: 0, successfulTasks: 0, failedTasks: 0, successRate: 0, avgResponseTime: 0, avgTasksPerConversation: 0 },
            topAgents: [],
            hourlyUsage: []
          })
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error)
        // Set empty metrics on error
        setMetrics({
          totalConversations: 0,
          totalWorkflows: 0,
          totalAgents: 0,
          recentActivity: 0,
          conversationTrend: [],
          workflowStatus: [],
          agentUsage: [],
          recentConversations: [],
          costMetrics: { today: 0, week: 0, month: 0, total: 0, avgPerConversation: 0 },
          costTrend: [],
          performanceMetrics: { totalTasks: 0, successfulTasks: 0, failedTasks: 0, successRate: 0, avgResponseTime: 0, avgTasksPerConversation: 0 },
          topAgents: [],
          hourlyUsage: []
        })
      } finally {
        setLoading(false)
      }
    }

    fetchMetrics()
  }, [user])

  if (loading) {
    return (
      <SidebarInset className={!open ? "ml-16" : ""}>
            <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
              <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
                <div className="flex items-center space-x-4">
                  <SidebarTrigger />
                </div>
              </div>
              <main className="p-6">
                <div className="flex items-center justify-center h-64">
                  <div className="text-center">
                    <Activity className="w-12 h-12 animate-spin text-blue-600 mx-auto mb-4" />
                    <p className="text-gray-600 dark:text-gray-400">Loading metrics...</p>
                  </div>
                </div>
              </main>
            </div>
          </SidebarInset>
    )
  }

  return (
    <SidebarInset className={!open ? "ml-16" : ""}>
          <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
            {/* Header */}
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
              <div className="flex items-center space-x-4">
                <SidebarTrigger />
              </div>
            </div>

            {/* Main Content */}
            <main className="p-6">
              {/* Title Section */}
              <div className="mb-6">
                <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-400">Metrics & Dashboard</h1>
                <p className="text-gray-600 dark:text-gray-300 mt-2">
                  Track your orchestration activity and performance
                </p>
              </div>

              {/* Key Metrics Cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Total Conversations</CardTitle>
                    <MessageSquare className="h-4 w-4 text-blue-600" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{metrics?.totalConversations || 0}</div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      {metrics?.recentActivity || 0} in last 24h
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Saved Workflows</CardTitle>
                    <Workflow className="h-4 w-4 text-green-600" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{metrics?.totalWorkflows || 0}</div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Reusable templates
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Available Agents</CardTitle>
                    <Users className="h-4 w-4 text-purple-600" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">{metrics?.totalAgents || 0}</div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      In directory
                    </p>
                  </CardContent>
                </Card>

                <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Activity Score</CardTitle>
                    <TrendingUp className="h-4 w-4 text-orange-600" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {metrics ? Math.min(100, (metrics.totalConversations * 10 + metrics.totalWorkflows * 20)) : 0}
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Based on usage
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Tabs for different views */}
              <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
                <TabsList className="mb-6">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="cost">Cost Analytics</TabsTrigger>
                  <TabsTrigger value="performance">Performance</TabsTrigger>
                  <TabsTrigger value="agents">Agents</TabsTrigger>
                </TabsList>

                <TabsContent value="overview" className="space-y-6">
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {/* Conversation Trend */}
                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader>
                        <CardTitle>Conversation Trend</CardTitle>
                        <CardDescription>Last 7 days activity</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                          <LineChart data={metrics?.conversationTrend || []}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="date" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line 
                              type="monotone" 
                              dataKey="count" 
                              stroke="#3b82f6" 
                              strokeWidth={2}
                              name="Conversations"
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>

                    {/* Workflow Status */}
                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader>
                        <CardTitle>Workflow Status</CardTitle>
                        <CardDescription>Distribution of workflow states</CardDescription>
                      </CardHeader>
                      <CardContent>
                        {metrics?.workflowStatus && metrics.workflowStatus.length > 0 ? (
                          <ResponsiveContainer width="100%" height={300}>
                            <PieChart>
                              <Pie
                                data={metrics.workflowStatus}
                                cx="50%"
                                cy="50%"
                                labelLine={false}
                                label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                                outerRadius={80}
                                fill="#8884d8"
                                dataKey="value"
                              >
                                {metrics.workflowStatus.map((entry, index) => (
                                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                                ))}
                              </Pie>
                              <Tooltip />
                            </PieChart>
                          </ResponsiveContainer>
                        ) : (
                          <div className="flex items-center justify-center h-[300px] text-gray-500">
                            No workflows yet
                          </div>
                        )}
                      </CardContent>
                    </Card>
                  </div>

                  {/* Recent Activity */}
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Recent Conversations</CardTitle>
                      <CardDescription>Your latest orchestration sessions</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {metrics?.recentConversations && metrics.recentConversations.length > 0 ? (
                        <div className="space-y-3">
                          {metrics.recentConversations.map((conv) => (
                            <div 
                              key={conv.id}
                              className="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
                            >
                              <div className="flex items-center space-x-3">
                                <CheckCircle className="w-5 h-5 text-green-600" />
                                <div>
                                  <p className="font-medium">{conv.title}</p>
                                  <p className="text-sm text-gray-600 dark:text-gray-400">{conv.date}</p>
                                </div>
                              </div>
                              <span className="text-xs px-2 py-1 rounded-full bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200">
                                {conv.status}
                              </span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          No conversations yet. Start a new orchestration to see activity here.
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="cost" className="space-y-6">
                  {/* Cost Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Today</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <DollarSign className="w-4 h-4 text-green-600 mr-1" />
                          <span className="text-2xl font-bold">{(metrics?.costMetrics?.today ?? 0).toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">This Week</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <DollarSign className="w-4 h-4 text-blue-600 mr-1" />
                          <span className="text-2xl font-bold">{(metrics?.costMetrics?.week ?? 0).toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">This Month</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <DollarSign className="w-4 h-4 text-purple-600 mr-1" />
                          <span className="text-2xl font-bold">{(metrics?.costMetrics?.month ?? 0).toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Total</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <DollarSign className="w-4 h-4 text-orange-600 mr-1" />
                          <span className="text-2xl font-bold">{(metrics?.costMetrics?.total ?? 0).toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Cost Trend Chart */}
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Cost Trend</CardTitle>
                      <CardDescription>Daily cost over the last 7 days</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={metrics?.costTrend || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="date" />
                          <YAxis />
                          <Tooltip formatter={(value: any) => `$${Number(value).toFixed(4)}`} />
                          <Legend />
                          <Line 
                            type="monotone" 
                            dataKey="cost" 
                            stroke="#10b981" 
                            strokeWidth={2}
                            name="Cost (USD)"
                          />
                        </LineChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Top Agents by Cost */}
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Top Agents by Cost</CardTitle>
                      <CardDescription>Most expensive agents in your workflows</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {metrics?.topAgents && metrics.topAgents.length > 0 ? (
                        <div className="space-y-3">
                          {metrics.topAgents.map((agent, index) => (
                            <div 
                              key={index}
                              className="flex items-center justify-between p-3 rounded-lg border border-gray-200 dark:border-gray-700"
                            >
                              <div>
                                <p className="font-medium">{agent.name}</p>
                                <p className="text-sm text-gray-600 dark:text-gray-400">
                                  {agent.calls} calls • ${(agent.costPerCall ?? 0).toFixed(4)} per call
                                </p>
                              </div>
                              <div className="text-right">
                                <p className="font-bold text-green-600">${(agent.cost ?? 0).toFixed(4)}</p>
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          No agent usage data available yet
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="performance" className="space-y-6">
                  {/* Performance Summary Cards */}
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Success Rate</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <Target className="w-4 h-4 text-green-600 mr-2" />
                          <span className="text-2xl font-bold">{(metrics?.performanceMetrics?.successRate ?? 0).toFixed(1)}%</span>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          {metrics?.performanceMetrics.successfulTasks || 0} of {metrics?.performanceMetrics.totalTasks || 0} tasks
                        </p>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Response Time</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <Zap className="w-4 h-4 text-yellow-600 mr-2" />
                          <span className="text-2xl font-bold">{(metrics?.performanceMetrics?.avgResponseTime ?? 0).toFixed(1)}</span>
                          <span className="text-sm text-gray-600 dark:text-gray-400 ml-1">min</span>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          Per conversation
                        </p>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader className="pb-2">
                        <CardTitle className="text-sm font-medium text-gray-600 dark:text-gray-400">Avg Tasks</CardTitle>
                      </CardHeader>
                      <CardContent>
                        <div className="flex items-center">
                          <BarChart3 className="w-4 h-4 text-purple-600 mr-2" />
                          <span className="text-2xl font-bold">{(metrics?.performanceMetrics?.avgTasksPerConversation ?? 0).toFixed(1)}</span>
                        </div>
                        <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                          Per conversation
                        </p>
                      </CardContent>
                    </Card>
                  </div>

                  {/* Hourly Usage Pattern */}
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Usage Pattern</CardTitle>
                      <CardDescription>Hourly distribution of conversations</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={metrics?.hourlyUsage || []}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="hour" />
                          <YAxis />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="count" fill="#3b82f6" name="Conversations" />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Task Success Breakdown */}
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader>
                        <CardTitle>Task Completion</CardTitle>
                        <CardDescription>Success vs failure breakdown</CardDescription>
                      </CardHeader>
                      <CardContent>
                        <ResponsiveContainer width="100%" height={250}>
                          <PieChart>
                            <Pie
                              data={[
                                { name: 'Successful', value: metrics?.performanceMetrics.successfulTasks || 0 },
                                { name: 'Failed', value: metrics?.performanceMetrics.failedTasks || 0 }
                              ]}
                              cx="50%"
                              cy="50%"
                              labelLine={false}
                              label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                              outerRadius={80}
                              fill="#8884d8"
                              dataKey="value"
                            >
                              <Cell fill="#10b981" />
                              <Cell fill="#ef4444" />
                            </Pie>
                            <Tooltip />
                          </PieChart>
                        </ResponsiveContainer>
                      </CardContent>
                    </Card>

                    <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                      <CardHeader>
                        <CardTitle>Performance Insights</CardTitle>
                        <CardDescription>Key performance indicators</CardDescription>
                      </CardHeader>
                      <CardContent className="space-y-4">
                        <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
                          <span className="text-sm font-medium">Total Tasks Executed</span>
                          <span className="text-lg font-bold">{metrics?.performanceMetrics.totalTasks || 0}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
                          <span className="text-sm font-medium">Successful Tasks</span>
                          <span className="text-lg font-bold text-green-600">{metrics?.performanceMetrics.successfulTasks || 0}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
                          <span className="text-sm font-medium">Failed Tasks</span>
                          <span className="text-lg font-bold text-red-600">{metrics?.performanceMetrics.failedTasks || 0}</span>
                        </div>
                        <div className="flex justify-between items-center p-3 rounded-lg bg-gray-50 dark:bg-gray-900">
                          <span className="text-sm font-medium">Avg Cost per Conversation</span>
                          <span className="text-lg font-bold text-blue-600">${(metrics?.costMetrics?.avgPerConversation ?? 0).toFixed(4)}</span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </TabsContent>

                <TabsContent value="conversations">
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Conversation Analytics</CardTitle>
                      <CardDescription>Detailed conversation metrics and insights</CardDescription>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-6">
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                          <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                            <div className="flex items-center space-x-2 mb-2">
                              <Clock className="w-4 h-4 text-blue-600" />
                              <span className="text-sm font-medium">Avg. Duration</span>
                            </div>
                            <p className="text-2xl font-bold">2.5 min</p>
                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Per conversation</p>
                          </div>
                          
                          <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                            <div className="flex items-center space-x-2 mb-2">
                              <CheckCircle className="w-4 h-4 text-green-600" />
                              <span className="text-sm font-medium">Success Rate</span>
                            </div>
                            <p className="text-2xl font-bold">94%</p>
                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Completed successfully</p>
                          </div>
                          
                          <div className="p-4 rounded-lg border border-gray-200 dark:border-gray-700">
                            <div className="flex items-center space-x-2 mb-2">
                              <BarChart3 className="w-4 h-4 text-purple-600" />
                              <span className="text-sm font-medium">Avg. Tasks</span>
                            </div>
                            <p className="text-2xl font-bold">3.2</p>
                            <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">Per conversation</p>
                          </div>
                        </div>

                        <div>
                          <h3 className="text-lg font-semibold mb-4">Conversation History</h3>
                          <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={metrics?.conversationTrend || []}>
                              <CartesianGrid strokeDasharray="3 3" />
                              <XAxis dataKey="date" />
                              <YAxis />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="count" fill="#3b82f6" name="Conversations" />
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="agents">
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Agent Usage Statistics</CardTitle>
                      <CardDescription>Most frequently used agents in your workflows</CardDescription>
                    </CardHeader>
                    <CardContent>
                      {metrics?.agentUsage && metrics.agentUsage.length > 0 ? (
                        <ResponsiveContainer width="100%" height={400}>
                          <BarChart data={metrics.agentUsage} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis type="number" />
                            <YAxis dataKey="name" type="category" width={150} />
                            <Tooltip />
                            <Legend />
                            <Bar dataKey="calls" fill="#8b5cf6" name="API Calls" />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="text-center py-8 text-gray-500">
                          No agent usage data available yet
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </main>
          </div>
        </SidebarInset>
  )
}

export default function MetricsPage() {
  return (
    <>
      <Navbar />
      <SidebarProvider>
        <AppSidebar />
        <MetricsContent />
      </SidebarProvider>
    </>
  )
}
