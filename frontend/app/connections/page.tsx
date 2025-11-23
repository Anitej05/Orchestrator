"use client"

import { useState, useEffect } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Loader2, Plus, Link as LinkIcon, Trash2, CheckCircle2, AlertCircle, Plug } from "lucide-react"
import { useUser } from "@clerk/nextjs"
import { useToast } from "@/hooks/use-toast"

interface ConnectionInfo {
  agent_id: string
  name: string
  description: string | null
  url: string | null
  tool_count: number
  tools: string[]
  created_at: string | null
}

interface Integration {
  id: string
  name: string
  description: string
  type: string
  url: string
  auth_mode: string
  fields?: Array<{
    key: string
    label: string
    type: string
    help: string
  }>
  icon?: string
}

export default function ConnectionsPage() {
  const { user } = useUser()
  const { toast } = useToast()
  
  const [url, setUrl] = useState("")
  const [status, setStatus] = useState<"idle" | "probing" | "auth_needed" | "connecting" | "success">("idle")
  const [authFields, setAuthFields] = useState<Array<{ key: string; label: string; type: string; help: string }>>([])
  const [credentials, setCredentials] = useState<Record<string, string>>({})
  const [connections, setConnections] = useState<ConnectionInfo[]>([])
  const [integrations, setIntegrations] = useState<Integration[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedIntegration, setSelectedIntegration] = useState<Integration | null>(null)

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  // Load connections and integrations on mount
  useEffect(() => {
    if (user) {
      loadConnections()
      loadIntegrations()
    }
  }, [user])

  const loadConnections = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/connect/list?user_id=${user?.id}`)
      if (response.ok) {
        const data = await response.json()
        setConnections(data)
      }
    } catch (error) {
      console.error("Failed to load connections:", error)
    } finally {
      setLoading(false)
    }
  }

  const loadIntegrations = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/connect/integrations`)
      if (response.ok) {
        const data = await response.json()
        setIntegrations(data)
      }
    } catch (error) {
      console.error("Failed to load integrations:", error)
    }
  }

  const handleProbe = async () => {
    if (!url) {
      toast({
        title: "Error",
        description: "Please enter a URL",
        variant: "destructive"
      })
      return
    }

    setStatus("probing")
    try {
      const response = await fetch(`${API_BASE}/api/connect/probe`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url })
      })

      const data = await response.json()

      if (data.status === "auth_required") {
        setStatus("auth_needed")
        // Determine fields based on probe result
        if (data.type === "api_key") {
          setAuthFields([{
            key: data.header || "Authorization",
            label: "API Key",
            type: "password",
            help: "Enter your API key for authentication"
          }])
        }
      } else if (data.status === "open") {
        // No auth needed, connect directly
        handleConnect({})
      } else {
        toast({
          title: "Connection Failed",
          description: data.message || "Could not connect to server",
          variant: "destructive"
        })
        setStatus("idle")
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to probe server",
        variant: "destructive"
      })
      setStatus("idle")
    }
  }

  const handleConnect = async (creds: Record<string, string>) => {
    setStatus("connecting")
    try {
      const response = await fetch(`${API_BASE}/api/connect/ingest`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          url,
          credentials: creds,
          user_id: user?.id
        })
      })

      const data = await response.json()

      if (data.status === "success") {
        setStatus("success")
        toast({
          title: "Success",
          description: `Connected to ${data.agent_name} with ${data.tool_count} tools`
        })
        
        // Reset form and reload connections
        setTimeout(() => {
          setStatus("idle")
          setUrl("")
          setCredentials({})
          setAuthFields([])
          setSelectedIntegration(null)
          loadConnections()
        }, 2000)
      } else {
        toast({
          title: "Connection Failed",
          description: data.message || "Failed to connect",
          variant: "destructive"
        })
        setStatus("idle")
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to connect to server",
        variant: "destructive"
      })
      setStatus("idle")
    }
  }

  const handleDelete = async (agentId: string) => {
    try {
      const response = await fetch(`${API_BASE}/api/connect/${agentId}?user_id=${user?.id}`, {
        method: "DELETE"
      })

      if (response.ok) {
        toast({
          title: "Success",
          description: "Connection deleted"
        })
        loadConnections()
      } else {
        toast({
          title: "Error",
          description: "Failed to delete connection",
          variant: "destructive"
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete connection",
        variant: "destructive"
      })
    }
  }

  const selectIntegration = (integration: Integration) => {
    setSelectedIntegration(integration)
    if (integration.url) {
      setUrl(integration.url)
    }
    if (integration.fields) {
      setAuthFields(integration.fields)
      setStatus("auth_needed")
    }
  }

  return (
    <div className="container mx-auto p-8 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Connections</h1>
        <p className="text-muted-foreground">
          Connect to MCP servers to extend your agent capabilities
        </p>
      </div>

      {/* Quick Connect Templates */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Quick Connect</CardTitle>
          <CardDescription>Connect to popular services with one click</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {integrations.map((integration) => (
              <Card
                key={integration.id}
                className="cursor-pointer hover:border-primary transition-colors"
                onClick={() => selectIntegration(integration)}
              >
                <CardContent className="pt-6">
                  <div className="flex items-start gap-3">
                    <div className="p-2 bg-primary/10 rounded-lg">
                      <Plug className="h-5 w-5 text-primary" />
                    </div>
                    <div className="flex-1">
                      <h3 className="font-semibold mb-1">{integration.name}</h3>
                      <p className="text-sm text-muted-foreground">{integration.description}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Add Connection Card */}
      <Card className="mb-8">
        <CardHeader>
          <CardTitle>Add New Connection</CardTitle>
          <CardDescription>
            {selectedIntegration
              ? `Connecting to ${selectedIntegration.name}`
              : "Enter an MCP server URL to connect"}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex gap-4">
              <Input
                placeholder="Enter MCP Server URL (e.g., https://mcp.example.com)"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                disabled={status !== "idle"}
              />
              <Button
                onClick={handleProbe}
                disabled={status === "probing" || !url}
              >
                {status === "probing" ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  "Detect"
                )}
              </Button>
            </div>

            {/* Dynamic Auth Form */}
            {status === "auth_needed" && (
              <div className="p-4 bg-muted rounded-lg space-y-4">
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <AlertCircle className="h-4 w-4" />
                  <span>This server requires authentication</span>
                </div>
                
                {authFields.map((field) => (
                  <div key={field.key} className="space-y-2">
                    <label className="text-sm font-medium">{field.label}</label>
                    <Input
                      type={field.type}
                      placeholder={field.help}
                      onChange={(e) =>
                        setCredentials({ ...credentials, [field.key]: e.target.value })
                      }
                    />
                    <p className="text-xs text-muted-foreground">{field.help}</p>
                  </div>
                ))}
                
                <Button
                  className="w-full"
                  onClick={() => handleConnect(credentials)}
                  disabled={status === "connecting"}
                >
                  {status === "connecting" ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Connecting...
                    </>
                  ) : (
                    "Connect & Save"
                  )}
                </Button>
              </div>
            )}

            {status === "success" && (
              <div className="flex items-center gap-2 p-4 bg-green-50 dark:bg-green-950 text-green-700 dark:text-green-300 rounded-lg">
                <CheckCircle2 className="h-5 w-5" />
                <span className="font-medium">Successfully connected!</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      {/* Existing Connections */}
      <Card>
        <CardHeader>
          <CardTitle>Your Connections</CardTitle>
          <CardDescription>Manage your connected MCP servers</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-6 w-6 animate-spin" />
            </div>
          ) : connections.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <Plug className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No connections yet. Add your first connection above.</p>
            </div>
          ) : (
            <div className="space-y-4">
              {connections.map((conn) => (
                <Card key={conn.agent_id}>
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <h3 className="font-semibold">{conn.name}</h3>
                          <Badge variant="secondary">{conn.tool_count} tools</Badge>
                        </div>
                        {conn.description && (
                          <p className="text-sm text-muted-foreground mb-2">
                            {conn.description}
                          </p>
                        )}
                        {conn.url && (
                          <p className="text-xs text-muted-foreground flex items-center gap-1">
                            <LinkIcon className="h-3 w-3" />
                            {conn.url}
                          </p>
                        )}
                        <div className="flex flex-wrap gap-1 mt-2">
                          {conn.tools.slice(0, 5).map((tool) => (
                            <Badge key={tool} variant="outline" className="text-xs">
                              {tool}
                            </Badge>
                          ))}
                          {conn.tools.length > 5 && (
                            <Badge variant="outline" className="text-xs">
                              +{conn.tools.length - 5} more
                            </Badge>
                          )}
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDelete(conn.agent_id)}
                      >
                        <Trash2 className="h-4 w-4 text-destructive" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
