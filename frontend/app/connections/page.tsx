"use client"

import { useEffect, useState, useRef, useCallback } from "react"
import { SidebarInset } from "@/components/ui/sidebar"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Loader2, AlertCircle, CheckCircle2, BookOpen, Mail, RefreshCw, Pause, Play } from "lucide-react"
import { useUser } from "@clerk/nextjs"
import { useToast } from "@/hooks/use-toast"
import { authFetch } from "@/lib/auth-fetch"

function ConnectionsContent() {
  const { user } = useUser()
  const { toast } = useToast()
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)
  
  // Integration connections state (multi-app support)
  const [integrationStatus, setIntegrationStatus] = useState<Record<string, "unknown" | "connected" | "disconnected" | "disabled" | "pending" | "error">>({
    "zohobooks": "unknown", 
    "gmail": "unknown"
  })
  const [integrationLoading, setIntegrationLoading] = useState<Record<string, boolean>>({"zohobooks": false, "gmail": false})
  const [integrationConnectionId, setIntegrationConnectionId] = useState<Record<string, string | null>>({"zohobooks": null, "gmail": null})
  const [integrationError, setIntegrationError] = useState<Record<string, string | null>>({"zohobooks": null, "gmail": null})
  const [troubleshootingHint, setTroubleshootingHint] = useState<Record<string, string | null>>({"zohobooks": null, "gmail": null})

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  useEffect(() => {
    if (user) {
      loadIntegrationStatus()
    }
  }, [user])

  useEffect(() => {
    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current)
      }
    }
  }, [])

  const loadIntegrationStatus = async () => {
    if (!user?.id) return
    
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/status/${user.id}`)
      const data = await response.json()
      
      if (data.success) {
        const statusMap: Record<string, "connected" | "disconnected" | "disabled" | "pending"> = {}
        const connectionIdMap: Record<string, string | null> = {}
        const errorMap: Record<string, string | null> = {}
        const hintMap: Record<string, string | null> = {}
        
        data.all_toolkits?.forEach((toolkit: any) => {
          if (toolkit.db_status === "disabled") {
            statusMap[toolkit.slug] = "disabled"
          } else if (toolkit.db_status === "active") {
            statusMap[toolkit.slug] = "connected"
          } else if (toolkit.db_status === "INITIATED") {
            statusMap[toolkit.slug] = "pending"
          } else if (toolkit.is_connected) {
            statusMap[toolkit.slug] = "connected"
          } else {
            statusMap[toolkit.slug] = "disconnected"
          }

          connectionIdMap[toolkit.slug] = toolkit.db_connection_id || toolkit.connected_account_id || null
          errorMap[toolkit.slug] = toolkit.error || null
          hintMap[toolkit.slug] = toolkit.troubleshooting || null
        })
        
        setIntegrationStatus(statusMap)
        setIntegrationConnectionId(connectionIdMap)
        setIntegrationError(errorMap)
        setTroubleshootingHint(hintMap)
      } else {
        setIntegrationError({"zohobooks": data.error || "Failed to check status", "gmail": data.error || "Failed to check status"})
      }
    } catch (error) {
      console.error("Error checking integration status:", error)
      setIntegrationError({"zohobooks": "Failed to check connection status", "gmail": "Failed to check connection status"})
      setIntegrationStatus({"zohobooks": "error", "gmail": "error"})
    }
  }

  const handleIntegrationConnect = async (appSlug: string) => {
    if (!user?.id) {
      toast({
        title: "Error",
        description: "User not authenticated",
        variant: "destructive"
      })
      return
    }

    setIntegrationLoading(prev => ({...prev, [appSlug]: true}))
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/auth/start/${user.id}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          app_slug: appSlug,
          callback_url: `${window.location.origin}/connections`
        })
      })
      const data = await response.json()

      if (data.success && data.redirect_url) {
        // Open OAuth in popup window
        const popup = window.open(data.redirect_url, "_blank", "width=600,height=700")
        
        toast({
          title: "Connecting",
          description: `Complete authentication in the popup window`
        })
        
        let pollCount = 0
        const maxPolls = 20
        
        pollIntervalRef.current = setInterval(async () => {
          pollCount++
          
          try {
            await authFetch(`${API_BASE}/api/integrations/sync/${user.id}`, {
              method: "POST"
            })
          } catch (syncError) {
            console.error("Sync failed:", syncError)
          }
          
          const statusResponse = await authFetch(`${API_BASE}/api/integrations/status/${user.id}/${appSlug}`)
          if (statusResponse.ok) {
            const statusData = await statusResponse.json()
            
            if (statusData.connected_apps && statusData.connected_apps.includes(appSlug)) {
              if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
              if (popup && !popup.closed) popup.close()
              
              setIntegrationStatus(prev => ({...prev, [appSlug]: "connected"}))
              setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
              
              toast({
                title: "Connected!",
                description: `Successfully connected to ${appSlug}`
              })
              
              loadIntegrationStatus()
              return
            }
          }
          
          if (popup && popup.closed) {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
            
            try {
              await authFetch(`${API_BASE}/api/integrations/sync/${user.id}`, {
                method: "POST"
              })
            } catch (syncError) {
              console.error("Final sync failed:", syncError)
            }
            loadIntegrationStatus()
          }
          
          if (pollCount >= maxPolls) {
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current)
            setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
            
            toast({
              title: "Timeout",
              description: "Connection took too long. Please check status manually.",
              variant: "destructive"
            })
          }
        }, 3000)
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to get connection URL",
          variant: "destructive"
        })
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to initiate connection",
        variant: "destructive"
      })
    } finally {
      setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
    }
  }

  const handleIntegrationDisconnect = async (appSlug: string) => {
    if (!user?.id) return

    setIntegrationLoading(prev => ({...prev, [appSlug]: true}))
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/disconnect/${user.id}/${appSlug}`, {
        method: "DELETE"
      })
      const data = await response.json()

      if (data.success) {
        toast({
          title: "Success",
          description: `${appSlug} disconnected`
        })
        setIntegrationStatus(prev => ({...prev, [appSlug]: "disconnected"}))
        setIntegrationConnectionId(prev => ({...prev, [appSlug]: null}))
        setIntegrationError(prev => ({...prev, [appSlug]: null}))
        setTroubleshootingHint(prev => ({...prev, [appSlug]: null}))
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to disconnect",
          variant: "destructive"
        })
        if (data.troubleshooting) {
          setTroubleshootingHint(prev => ({...prev, [appSlug]: data.troubleshooting}))
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to disconnect",
        variant: "destructive"
      })
    } finally {
      setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
    }
  }

  const handleIntegrationRefresh = async (appSlug: string) => {
    if (!user?.id) return

    setIntegrationLoading(prev => ({...prev, [appSlug]: true}))
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/refresh/${user.id}/${appSlug}`, {
        method: "POST"
      })
      const data = await response.json()

      if (data.success) {
        toast({
          title: "Success",
          description: `${appSlug} connection refreshed`
        })
        loadIntegrationStatus()
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to refresh connection",
          variant: "destructive"
        })
        if (data.troubleshooting) {
          setTroubleshootingHint(prev => ({...prev, [appSlug]: data.troubleshooting}))
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to refresh connection",
        variant: "destructive"
      })
    } finally {
      setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
    }
  }

  const handleIntegrationDisable = async (appSlug: string) => {
    if (!user?.id) return

    setIntegrationLoading(prev => ({...prev, [appSlug]: true}))
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/disable/${user.id}/${appSlug}`, {
        method: "POST"
      })
      const data = await response.json()

      if (data.success) {
        toast({
          title: "Success",
          description: `${appSlug} connection paused`
        })
        setIntegrationStatus(prev => ({...prev, [appSlug]: "disabled"}))
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to disable connection",
          variant: "destructive"
        })
        if (data.troubleshooting) {
          setTroubleshootingHint(prev => ({...prev, [appSlug]: data.troubleshooting}))
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to disable connection",
        variant: "destructive"
      })
    } finally {
      setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
    }
  }

  const handleIntegrationEnable = async (appSlug: string) => {
    if (!user?.id) return

    setIntegrationLoading(prev => ({...prev, [appSlug]: true}))
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/enable/${user.id}/${appSlug}`, {
        method: "POST"
      })
      const data = await response.json()

      if (data.success) {
        toast({
          title: "Success",
          description: `${appSlug} connection resumed`
        })
        setIntegrationStatus(prev => ({...prev, [appSlug]: "connected"}))
      } else {
        toast({
          title: "Error",
          description: data.error || "Failed to enable connection",
          variant: "destructive"
        })
        if (data.troubleshooting) {
          setTroubleshootingHint(prev => ({...prev, [appSlug]: data.troubleshooting}))
        }
      }
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to enable connection",
        variant: "destructive"
      })
    } finally {
      setIntegrationLoading(prev => ({...prev, [appSlug]: false}))
    }
  }



  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        <main className="p-6 space-y-6">
          <div>
            <h1 className="text-2xl font-semibold text-brand-teal">Connections</h1>
            <p className="text-text-secondary text-sm">Connect to MCP servers to extend your agent capabilities</p>
          </div>
          <Card className="ui-card">
            <CardHeader>
              <CardTitle className="text-text-primary">Connected Applications</CardTitle>
              <CardDescription className="text-text-secondary">Manage OAuth connections for external services</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Zoho Books Card */}
                <Card className="ui-card border-l-4 border-l-brand-teal">
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-3 flex-1">
                        <div className="p-2 rounded-lg bg-brand-teal/10 text-brand-teal">
                          <BookOpen className="h-5 w-5" />
                        </div>
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center gap-2">
                            <h3 className="font-semibold text-text-primary">Zoho Books</h3>
                            {integrationLoading["zohobooks"] ? (
                              <Loader2 className="h-4 w-4 animate-spin text-text-secondary" />
                            ) : (integrationStatus["zohobooks"] ?? "disconnected") === "connected" ? (
                              <Badge className="bg-status-success/15 text-status-success border border-status-success/30">
                                <CheckCircle2 className="h-3 w-3 mr-1" />
                                Connected
                              </Badge>
                            ) : (integrationStatus["zohobooks"] ?? "disconnected") === "disabled" ? (
                              <Badge className="bg-yellow-500/15 text-yellow-600 border border-yellow-500/30">
                                <Pause className="h-3 w-3 mr-1" />
                                Paused
                              </Badge>
                            ) : (integrationStatus["zohobooks"] ?? "disconnected") === "pending" ? (
                              <Badge className="bg-status-warning/15 text-status-warning border border-status-warning/30">
                                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                                Connecting
                              </Badge>
                            ) : (integrationStatus["zohobooks"] ?? "disconnected") === "disconnected" ? (
                              <Badge variant="outline" className="border-border-color text-text-secondary">
                                Not Connected
                              </Badge>
                            ) : (integrationStatus["zohobooks"] ?? "disconnected") === "error" ? (
                              <Badge className="bg-status-error/15 text-status-error border border-status-error/30">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                Error
                              </Badge>
                            ) : null}
                          </div>
                          <p className="text-sm text-text-secondary">
                            Create, list, update, and send invoices
                          </p>
                          {integrationError["zohobooks"] && (
                            <p className="text-xs text-status-error">{integrationError["zohobooks"]}</p>
                          )}
                          {troubleshootingHint["zohobooks"] && (
                            <p className="text-xs text-blue-600 dark:text-blue-400">💡 {troubleshootingHint["zohobooks"]}</p>
                          )}
                          {integrationConnectionId["zohobooks"] && (
                            <p className="text-xs text-text-secondary">ID: {integrationConnectionId["zohobooks"]}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {(integrationStatus["zohobooks"] ?? "disconnected") === "connected" ? (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationRefresh("zohobooks")}
                              disabled={integrationLoading["zohobooks"]}
                              title="Refresh connection"
                            >
                              <RefreshCw className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationDisable("zohobooks")}
                              disabled={integrationLoading["zohobooks"]}
                              title="Pause connection"
                            >
                              <Pause className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleIntegrationDisconnect("zohobooks")}
                              disabled={integrationLoading["zohobooks"]}
                            >
                              {integrationLoading["zohobooks"] ? <Loader2 className="h-4 w-4 animate-spin" /> : "Disconnect"}
                            </Button>
                          </>
                        ) : (integrationStatus["zohobooks"] ?? "disconnected") === "disabled" ? (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationEnable("zohobooks")}
                              disabled={integrationLoading["zohobooks"]}
                              title="Resume connection"
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleIntegrationDisconnect("zohobooks")}
                              disabled={integrationLoading["zohobooks"]}
                            >
                              {integrationLoading["zohobooks"] ? <Loader2 className="h-4 w-4 animate-spin" /> : "Disconnect"}
                            </Button>
                          </>
                        ) : (
                          <Button
                            size="sm"
                            onClick={() => handleIntegrationConnect("zohobooks")}
                            disabled={integrationLoading["zohobooks"]}
                          >
                            {integrationLoading["zohobooks"] ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                            Connect
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Gmail Card */}
                <Card className="ui-card border-l-4 border-l-brand-teal">
                  <CardContent className="pt-6">
                    <div className="flex items-start justify-between gap-4">
                      <div className="flex items-start gap-3 flex-1">
                        <div className="p-2 rounded-lg bg-brand-teal/10 text-brand-teal">
                          <Mail className="h-5 w-5" />
                        </div>
                        <div className="flex-1 space-y-2">
                          <div className="flex items-center gap-2">
                            <h3 className="font-semibold text-text-primary">Gmail</h3>
                            {integrationLoading["gmail"] ? (
                              <Loader2 className="h-4 w-4 animate-spin text-text-secondary" />
                            ) : (integrationStatus["gmail"] ?? "disconnected") === "connected" ? (
                              <Badge className="bg-status-success/15 text-status-success border border-status-success/30">
                                <CheckCircle2 className="h-3 w-3 mr-1" />
                                Connected
                              </Badge>
                            ) : (integrationStatus["gmail"] ?? "disconnected") === "disabled" ? (
                              <Badge className="bg-yellow-500/15 text-yellow-600 border border-yellow-500/30">
                                <Pause className="h-3 w-3 mr-1" />
                                Paused
                              </Badge>
                            ) : (integrationStatus["gmail"] ?? "disconnected") === "pending" ? (
                              <Badge className="bg-status-warning/15 text-status-warning border border-status-warning/30">
                                <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                                Connecting
                              </Badge>
                            ) : (integrationStatus["gmail"] ?? "disconnected") === "disconnected" ? (
                              <Badge variant="outline" className="border-border-color text-text-secondary">
                                Not Connected
                              </Badge>
                            ) : (integrationStatus["gmail"] ?? "disconnected") === "error" ? (
                              <Badge className="bg-status-error/15 text-status-error border border-status-error/30">
                                <AlertCircle className="h-3 w-3 mr-1" />
                                Error
                              </Badge>
                            ) : null}
                          </div>
                          <p className="text-sm text-text-secondary">
                            Send emails and manage inbox
                          </p>
                          {integrationError["gmail"] && (
                            <p className="text-xs text-status-error">{integrationError["gmail"]}</p>
                          )}
                          {troubleshootingHint["gmail"] && (
                            <p className="text-xs text-blue-600 dark:text-blue-400">💡 {troubleshootingHint["gmail"]}</p>
                          )}
                          {integrationConnectionId["gmail"] && (
                            <p className="text-xs text-text-secondary">ID: {integrationConnectionId["gmail"]}</p>
                          )}
                        </div>
                      </div>
                      <div className="flex gap-2">
                        {(integrationStatus["gmail"] ?? "disconnected") === "connected" ? (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationRefresh("gmail")}
                              disabled={integrationLoading["gmail"]}
                              title="Refresh connection"
                            >
                              <RefreshCw className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationDisable("gmail")}
                              disabled={integrationLoading["gmail"]}
                              title="Pause connection"
                            >
                              <Pause className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleIntegrationDisconnect("gmail")}
                              disabled={integrationLoading["gmail"]}
                            >
                              {integrationLoading["gmail"] ? <Loader2 className="h-4 w-4 animate-spin" /> : "Disconnect"}
                            </Button>
                          </>
                        ) : (integrationStatus["gmail"] ?? "disconnected") === "disabled" ? (
                          <>
                            <Button
                              variant="ghost"
                              size="sm"
                              onClick={() => handleIntegrationEnable("gmail")}
                              disabled={integrationLoading["gmail"]}
                              title="Resume connection"
                            >
                              <Play className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="outline"
                              size="sm"
                              onClick={() => handleIntegrationDisconnect("gmail")}
                              disabled={integrationLoading["gmail"]}
                            >
                              {integrationLoading["gmail"] ? <Loader2 className="h-4 w-4 animate-spin" /> : "Disconnect"}
                            </Button>
                          </>
                        ) : (
                          <Button
                            size="sm"
                            onClick={() => handleIntegrationConnect("gmail")}
                            disabled={integrationLoading["gmail"]}
                          >
                            {integrationLoading["gmail"] ? <Loader2 className="h-4 w-4 animate-spin mr-2" /> : null}
                            Connect
                          </Button>
                        )}
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>
            </CardContent>
          </Card>


        </main>
      </div>
    </SidebarInset>
  )
}

export default function ConnectionsPage() {
  return (
    <>
      <ConnectionsContent />
    </>
  )
}


