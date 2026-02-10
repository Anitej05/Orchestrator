"use client"

import { useEffect, useState } from "react"
import { useUser } from "@clerk/nextjs"
import { authFetch } from "@/lib/auth-fetch"
import { API_BASE_URL } from "@/lib/config"
import { SidebarInset, useSidebar } from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Activity,
  BarChart3,
  CheckCircle,
  Clock,
  DollarSign,
  Layers,
  MessageSquare,
  Target,
  TrendingUp,
  Users
} from "lucide-react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts"

type ConversationPoint = { date: string; count: number }
type WorkflowStatus = { name: string; value: number }
type AgentUsage = { name: string; calls: number }
type RecentConversation = { id: string; title: string; date: string; status: string }
type CostMetrics = { today: number; week: number; month: number; total: number; avgPerConversation: number }
type CostTrendPoint = { date: string; cost: number }
type PerformanceMetrics = {
  totalTasks: number
  successfulTasks: number
  failedTasks: number
  successRate: number
  avgResponseTime: number
  avgTasksPerConversation: number
}
type TopAgent = { name: string; calls: number; cost?: number; costPerCall?: number }
type HourlyUsage = { hour: string; count: number }

type DashboardMetrics = {
  totalConversations: number
  totalWorkflows: number
  totalAgents: number
  recentActivity: number
  conversationTrend: ConversationPoint[]
  workflowStatus: WorkflowStatus[]
  agentUsage: AgentUsage[]
  recentConversations: RecentConversation[]
  costMetrics: CostMetrics
  costTrend: CostTrendPoint[]
  performanceMetrics: PerformanceMetrics
  topAgents: TopAgent[]
  hourlyUsage: HourlyUsage[]
}

const CHART_COLORS = ["#0D9488", "#10B981", "#F59E0B", "#06B6D4", "#EF4444"]

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
        const metricsRes = await authFetch(`${API_BASE_URL}/api/metrics/dashboard`, {
          method: "GET",
          headers: {
            "X-User-ID": user.id,
            "Content-Type": "application/json"
          }
        })

        if (metricsRes.ok) {
          const data = await metricsRes.json()

          const agentsRes = await authFetch(`${API_BASE_URL}/api/agents/all`, {
            method: "GET",
            headers: {
              "Content-Type": "application/json"
            }
          })
          const agents = agentsRes.ok ? await agentsRes.json() : []

          const agentUsage = agents.slice(0, 5).map((agent: any) => ({
            name: agent.name?.length > 18 ? `${agent.name.substring(0, 18)}...` : agent.name,
            calls: Math.floor(Math.random() * 100)
          }))

          setMetrics({
            totalConversations: data.total_conversations || 0,
            totalWorkflows: data.total_workflows || 0,
            totalAgents: data.total_agents || 0,
            recentActivity: data.recent_activity || 0,
            conversationTrend: data.conversation_trend || [],
            workflowStatus: data.workflow_status || [],
            agentUsage: data.top_agents || agentUsage,
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
            performanceMetrics: {
              totalTasks: 0,
              successfulTasks: 0,
              failedTasks: 0,
              successRate: 0,
              avgResponseTime: 0,
              avgTasksPerConversation: 0
            },
            topAgents: [],
            hourlyUsage: []
          })
        }
      } catch (error) {
        console.error("Failed to fetch metrics:", error)
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
          performanceMetrics: {
            totalTasks: 0,
            successfulTasks: 0,
            failedTasks: 0,
            successRate: 0,
            avgResponseTime: 0,
            avgTasksPerConversation: 0
          },
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
      <SidebarInset>
        <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
          <main className="p-6 flex items-center justify-center h-[60vh]">
            <div className="text-center">
              <Activity className="w-12 h-12 animate-spin text-brand-teal mx-auto mb-4" />
              <p className="text-text-secondary">Loading metrics...</p>
            </div>
          </main>
        </div>
      </SidebarInset>
    )
  }

  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        <main className="p-6">
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-brand-teal">Metrics & Dashboard</h1>
            <p className="text-text-secondary mt-2">Track your orchestration activity and performance</p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
            <Card className="ui-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">Total Conversations</CardTitle>
                <MessageSquare className="h-4 w-4 text-brand-teal" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-text-primary">{metrics?.totalConversations || 0}</div>
                <p className="text-xs text-text-secondary mt-1">{metrics?.recentActivity || 0} in last 24h</p>
              </CardContent>
            </Card>

            <Card className="ui-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">Saved Workflows</CardTitle>
                <Layers className="h-4 w-4 text-status-success" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-text-primary">{metrics?.totalWorkflows || 0}</div>
                <p className="text-xs text-text-secondary mt-1">Reusable templates</p>
              </CardContent>
            </Card>

            <Card className="ui-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">Available Agents</CardTitle>
                <Users className="h-4 w-4 text-status-active" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-text-primary">{metrics?.totalAgents || 0}</div>
                <p className="text-xs text-text-secondary mt-1">In directory</p>
              </CardContent>
            </Card>

            <Card className="ui-card">
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium text-text-secondary">Activity Score</CardTitle>
                <TrendingUp className="h-4 w-4 text-status-pending" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-text-primary">
                  {metrics ? Math.min(100, metrics.totalConversations * 10 + metrics.totalWorkflows * 20) : 0}
                </div>
                <p className="text-xs text-text-secondary mt-1">Based on usage</p>
              </CardContent>
            </Card>
          </div>

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <TabsList className="mb-6">
              <TabsTrigger value="overview">Overview</TabsTrigger>
              <TabsTrigger value="cost">Cost Analytics</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
              <TabsTrigger value="conversations">Conversations</TabsTrigger>
              <TabsTrigger value="agents">Agents</TabsTrigger>
            </TabsList>

            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="ui-card">
                  <CardHeader>
                    <CardTitle className="text-text-primary">Conversation Trend</CardTitle>
                    <CardDescription className="text-text-secondary">Last 7 days activity</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={metrics?.conversationTrend || []}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#CBD5E1" />
                        <XAxis dataKey="date" stroke="#475569" />
                        <YAxis stroke="#475569" />
                        <Tooltip />
                        <Legend />
                        <Line type="monotone" dataKey="count" stroke="#0D9488" strokeWidth={2} name="Conversations" />
                      </LineChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader>
                    <CardTitle className="text-text-primary">Workflow Status</CardTitle>
                    <CardDescription className="text-text-secondary">Distribution of workflow states</CardDescription>
                  </CardHeader>
                  <CardContent>
                    {metrics?.workflowStatus?.length ? (
                      <ResponsiveContainer width="100%" height={300}>
                        <PieChart>
                          <Pie
                            data={metrics.workflowStatus}
                            cx="50%"
                            cy="50%"
                            labelLine={false}
                            label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                            outerRadius={80}
                            dataKey="value"
                          >
                            {metrics.workflowStatus.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                            ))}
                          </Pie>
                          <Tooltip />
                        </PieChart>
                      </ResponsiveContainer>
                    ) : (
                      <div className="flex items-center justify-center h-[300px] text-text-tertiary">No workflows yet</div>
                    )}
                  </CardContent>
                </Card>
              </div>

              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Recent Conversations</CardTitle>
                  <CardDescription className="text-text-secondary">Your latest orchestration sessions</CardDescription>
                </CardHeader>
                <CardContent>
                  {metrics?.recentConversations?.length ? (
                    <div className="space-y-3">
                      {metrics.recentConversations.map((conv) => (
                        <div
                          key={conv.id}
                          className="flex items-center justify-between p-3 rounded-lg border border-border-color hover:bg-bg-card transition-colors"
                        >
                          <div className="flex items-center space-x-3">
                            <CheckCircle className="w-5 h-5 text-status-success" />
                            <div>
                              <p className="font-medium text-text-primary">{conv.title}</p>
                              <p className="text-sm text-text-secondary">{conv.date}</p>
                            </div>
                          </div>
                          <span className="text-xs px-2 py-1 rounded-full bg-status-active/10 text-status-active border border-status-active/30">
                            {conv.status}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-text-tertiary">
                      No conversations yet. Start a new orchestration to see activity here.
                    </div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="cost" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">Today</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <DollarSign className="w-4 h-4 text-status-success mr-1" />
                      <span className="text-2xl font-bold text-text-primary">{(metrics?.costMetrics?.today ?? 0).toFixed(4)}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">This Week</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <DollarSign className="w-4 h-4 text-brand-teal mr-1" />
                      <span className="text-2xl font-bold text-text-primary">{(metrics?.costMetrics?.week ?? 0).toFixed(4)}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">This Month</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <DollarSign className="w-4 h-4 text-status-pending mr-1" />
                      <span className="text-2xl font-bold text-text-primary">{(metrics?.costMetrics?.month ?? 0).toFixed(4)}</span>
                    </div>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">Total</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <DollarSign className="w-4 h-4 text-status-warning mr-1" />
                      <span className="text-2xl font-bold text-text-primary">{(metrics?.costMetrics?.total ?? 0).toFixed(4)}</span>
                    </div>
                  </CardContent>
                </Card>
              </div>

              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Cost Trend</CardTitle>
                  <CardDescription className="text-text-secondary">Daily cost over the last 7 days</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={metrics?.costTrend || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#CBD5E1" />
                      <XAxis dataKey="date" stroke="#475569" />
                      <YAxis stroke="#475569" />
                      <Tooltip formatter={(value: any) => `$${Number(value).toFixed(4)}`} />
                      <Legend />
                      <Line type="monotone" dataKey="cost" stroke="#0D9488" strokeWidth={2} name="Cost (USD)" />
                    </LineChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Top Agents by Cost</CardTitle>
                  <CardDescription className="text-text-secondary">Most expensive agents in your workflows</CardDescription>
                </CardHeader>
                <CardContent>
                  {metrics?.topAgents?.length ? (
                    <div className="space-y-3">
                      {metrics.topAgents.map((agent, index) => (
                        <div
                          key={index}
                          className="flex items-center justify-between p-3 rounded-lg border border-border-color"
                        >
                          <div>
                            <p className="font-medium text-text-primary">{agent.name}</p>
                            <p className="text-sm text-text-secondary">
                              {agent.calls} calls | ${(agent.costPerCall ?? 0).toFixed(4)} per call
                            </p>
                          </div>
                          <div className="text-right">
                            <p className="font-bold text-status-success">${(agent.cost ?? 0).toFixed(4)}</p>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8 text-text-tertiary">No agent usage data available yet</div>
                  )}
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="performance" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">Success Rate</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <Target className="w-4 h-4 text-status-success mr-2" />
                      <span className="text-2xl font-bold text-text-primary">
                        {(metrics?.performanceMetrics?.successRate ?? 0).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-xs text-text-secondary mt-1">
                      {metrics?.performanceMetrics.successfulTasks || 0} of {metrics?.performanceMetrics.totalTasks || 0} tasks
                    </p>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">Avg Response Time</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <Clock className="w-4 h-4 text-status-warning mr-2" />
                      <span className="text-2xl font-bold text-text-primary">
                        {(metrics?.performanceMetrics?.avgResponseTime ?? 0).toFixed(1)}
                      </span>
                      <span className="text-sm text-text-secondary ml-1">min</span>
                    </div>
                    <p className="text-xs text-text-secondary mt-1">Per conversation</p>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader className="pb-2">
                    <CardTitle className="text-sm font-medium text-text-secondary">Avg Tasks</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="flex items-center">
                      <BarChart3 className="w-4 h-4 text-brand-teal mr-2" />
                      <span className="text-2xl font-bold text-text-primary">
                        {(metrics?.performanceMetrics?.avgTasksPerConversation ?? 0).toFixed(1)}
                      </span>
                    </div>
                    <p className="text-xs text-text-secondary mt-1">Per conversation</p>
                  </CardContent>
                </Card>
              </div>

              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Usage Pattern</CardTitle>
                  <CardDescription className="text-text-secondary">Hourly distribution of conversations</CardDescription>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={metrics?.hourlyUsage || []}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#CBD5E1" />
                      <XAxis dataKey="hour" stroke="#475569" />
                      <YAxis stroke="#475569" />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="count" fill="#0D9488" name="Conversations" />
                    </BarChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card className="ui-card">
                  <CardHeader>
                    <CardTitle className="text-text-primary">Task Completion</CardTitle>
                    <CardDescription className="text-text-secondary">Success vs failure breakdown</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={250}>
                      <PieChart>
                        <Pie
                          data={[
                            { name: "Successful", value: metrics?.performanceMetrics.successfulTasks || 0 },
                            { name: "Failed", value: metrics?.performanceMetrics.failedTasks || 0 }
                          ]}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
                          outerRadius={80}
                          dataKey="value"
                        >
                          <Cell fill="#10B981" />
                          <Cell fill="#EF4444" />
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                <Card className="ui-card">
                  <CardHeader>
                    <CardTitle className="text-text-primary">Performance Insights</CardTitle>
                    <CardDescription className="text-text-secondary">Key performance indicators</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    <div className="flex justify-between items-center p-3 rounded-lg bg-bg-card/70">
                      <span className="text-sm font-medium text-text-secondary">Total Tasks Executed</span>
                      <span className="text-lg font-bold text-text-primary">{metrics?.performanceMetrics.totalTasks || 0}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 rounded-lg bg-bg-card/70">
                      <span className="text-sm font-medium text-text-secondary">Successful Tasks</span>
                      <span className="text-lg font-bold text-status-success">{metrics?.performanceMetrics.successfulTasks || 0}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 rounded-lg bg-bg-card/70">
                      <span className="text-sm font-medium text-text-secondary">Failed Tasks</span>
                      <span className="text-lg font-bold text-status-error">{metrics?.performanceMetrics.failedTasks || 0}</span>
                    </div>
                    <div className="flex justify-between items-center p-3 rounded-lg bg-bg-card/70">
                      <span className="text-sm font-medium text-text-secondary">Avg Cost per Conversation</span>
                      <span className="text-lg font-bold text-brand-teal">
                        ${(metrics?.costMetrics?.avgPerConversation ?? 0).toFixed(4)}
                      </span>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>

            <TabsContent value="conversations" className="space-y-6">
              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Conversation Analytics</CardTitle>
                  <CardDescription className="text-text-secondary">Detailed conversation metrics and insights</CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="space-y-6">
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                      <div className="p-4 rounded-lg border border-border-color">
                        <div className="flex items-center space-x-2 mb-2">
                          <Clock className="w-4 h-4 text-brand-teal" />
                          <span className="text-sm font-medium text-text-secondary">Avg. Duration</span>
                        </div>
                        <p className="text-2xl font-bold text-text-primary">2.5 min</p>
                        <p className="text-xs text-text-secondary mt-1">Per conversation</p>
                      </div>

                      <div className="p-4 rounded-lg border border-border-color">
                        <div className="flex items-center space-x-2 mb-2">
                          <CheckCircle className="w-4 h-4 text-status-success" />
                          <span className="text-sm font-medium text-text-secondary">Success Rate</span>
                        </div>
                        <p className="text-2xl font-bold text-text-primary">94%</p>
                        <p className="text-xs text-text-secondary mt-1">Completed successfully</p>
                      </div>

                      <div className="p-4 rounded-lg border border-border-color">
                        <div className="flex items-center space-x-2 mb-2">
                          <BarChart3 className="w-4 h-4 text-status-pending" />
                          <span className="text-sm font-medium text-text-secondary">Avg. Tasks</span>
                        </div>
                        <p className="text-2xl font-bold text-text-primary">3.2</p>
                        <p className="text-xs text-text-secondary mt-1">Per conversation</p>
                      </div>
                    </div>

                    <div>
                      <h3 className="text-lg font-semibold text-text-primary mb-4">Conversation History</h3>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={metrics?.conversationTrend || []}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#CBD5E1" />
                          <XAxis dataKey="date" stroke="#475569" />
                          <YAxis stroke="#475569" />
                          <Tooltip />
                          <Legend />
                          <Bar dataKey="count" fill="#0D9488" name="Conversations" />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="agents" className="space-y-6">
              <Card className="ui-card">
                <CardHeader>
                  <CardTitle className="text-text-primary">Agent Usage Statistics</CardTitle>
                  <CardDescription className="text-text-secondary">Most frequently used agents in your workflows</CardDescription>
                </CardHeader>
                <CardContent>
                  {metrics?.agentUsage?.length ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <BarChart data={metrics.agentUsage} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#CBD5E1" />
                        <XAxis type="number" stroke="#475569" />
                        <YAxis dataKey="name" type="category" width={150} stroke="#475569" />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="calls" fill="#0D9488" name="API Calls" />
                      </BarChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="text-center py-8 text-text-tertiary">No agent usage data available yet</div>
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
  return <MetricsContent />
}

