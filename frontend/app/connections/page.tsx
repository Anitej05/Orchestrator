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
import { SearchBar } from "@/components/search-bar"
import { AppLogo } from "@/components/app-logo"

function ConnectionsContent() {
  const { user } = useUser()
  const { toast } = useToast()
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null)

  // Available toolkits from Composio (dynamically loaded)
  const [availableToolkits, setAvailableToolkits] = useState<any[]>([])
  const [filteredToolkits, setFilteredToolkits] = useState<any[]>([])
  const [toolkitsLoading, setToolkitsLoading] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  
  // Integration connections state (multi-app support)
  const [integrationStatus, setIntegrationStatus] = useState<Record<string, "unknown" | "connected" | "disconnected" | "disabled" | "pending" | "error">>({})
  const [integrationLoading, setIntegrationLoading] = useState<Record<string, boolean>>({})
  const [integrationConnectionId, setIntegrationConnectionId] = useState<Record<string, string | null>>({})
  const [integrationError, setIntegrationError] = useState<Record<string, string | null>>({})
  const [troubleshootingHint, setTroubleshootingHint] = useState<Record<string, string | null>>({})

  const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

  useEffect(() => {
    if (user) {
      loadAvailableToolkits()
      loadIntegrationStatus()
    }
  }, [user])

  // Filter toolkits based on search query
  useEffect(() => {
    if (searchQuery.trim() === "") {
      setFilteredToolkits(availableToolkits)
    } else {
      const query = searchQuery.toLowerCase()
      const filtered = availableToolkits.filter((toolkit) =>
        toolkit.name?.toLowerCase().includes(query) ||
        toolkit.slug?.toLowerCase().includes(query) ||
        toolkit.description?.toLowerCase().includes(query) ||
        toolkit.category?.toLowerCase().includes(query)
      )
      setFilteredToolkits(filtered)
    }
  }, [searchQuery, availableToolkits])

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
      }
    } catch (error) {
      console.error("Error checking integration status:", error)
    }
  }

  const loadAvailableToolkits = async () => {
    setToolkitsLoading(true)
    try {
      const response = await authFetch(`${API_BASE}/api/integrations/available`)
      const data = await response.json()

      if (data.success && data.toolkits) {
        setAvailableToolkits(data.toolkits)
        setFilteredToolkits(data.toolkits)

        // Initialize state for all toolkits
        const statusMap: Record<string, "unknown" | "connected" | "disconnected" | "disabled" | "pending" | "error"> = {}
        const loadingMap: Record<string, boolean> = {}
        const connectionIdMap: Record<string, string | null> = {}
        const errorMap: Record<string, string | null> = {}
        const hintMap: Record<string, string | null> = {}

        data.toolkits.forEach((toolkit: any) => {
          const slug = toolkit.slug || toolkit.name?.toLowerCase()
          if (slug) {
            statusMap[slug] = "unknown"
            loadingMap[slug] = false
            connectionIdMap[slug] = null
            errorMap[slug] = null
            hintMap[slug] = null
          }
        })

        setIntegrationStatus(prev => ({...prev, ...statusMap}))
        setIntegrationLoading(prev => ({...prev, ...loadingMap}))
        setIntegrationConnectionId(prev => ({...prev, ...connectionIdMap}))
        setIntegrationError(prev => ({...prev, ...errorMap}))
        setTroubleshootingHint(prev => ({...prev, ...hintMap}))
      }
    } catch (error) {
      console.error("Error loading available toolkits:", error)
      toast({
        title: "Error",
        description: "Failed to load available integrations",
        variant: "destructive"
      })
    } finally {
      setToolkitsLoading(false)
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
        
        let alreadyConnected = false  // guard: fire toast only once
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

            if (!alreadyConnected && statusData.connected_apps && statusData.connected_apps.includes(appSlug)) {
              alreadyConnected = true
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

  // Helper to get icon for common apps
  const getAppIcon = (slug: string) => {
    const iconMap: Record<string, any> = {
      "gmail": Mail,
      "zohobooks": BookOpen,
      "github": BookOpen,
      "slack": BookOpen,
      "notion": BookOpen,
    }
    return iconMap[slug] || BookOpen
  }

  // Get Composio app logo URL - GitHub raw CDN (ACTUAL working source)
  const getAppLogoUrl = (slug: string) => {
    // Composio hosts logos on GitHub: https://github.com/ComposioHQ/open-logos
    // Try PNG first, then SVG
    return {
      png: `https://raw.githubusercontent.com/ComposioHQ/open-logos/master/icons/${slug}.png`,
      svg: `https://raw.githubusercontent.com/ComposioHQ/open-logos/master/icons/${slug}.svg`,
    }
  }

  // Render toolkit card
  const renderToolkitCard = (toolkit: any) => {
    const slug = toolkit.slug || toolkit.name?.toLowerCase().replace(/\s+/g, "")
    const name = toolkit.name || slug
    const description = toolkit.description || toolkit.long_description || `Connect and automate workflows with ${name}`
    const status = integrationStatus[slug] ?? "disconnected"
    const loading = integrationLoading[slug] ?? false
    const error = integrationError[slug]
    const hint = troubleshootingHint[slug]

    return (
      <Card key={slug} className="ui-card border border-border-color hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
        <CardContent className="p-6">
          <div className="flex flex-col gap-4">
            {/* Logo with automatic fallback */}
            <AppLogo slug={slug} name={name} />
            
            {/* Name and Status */}
            <div className="text-center space-y-2">
              <h3 className="font-semibold text-lg text-text-primary line-clamp-1">{name}</h3>
              <div className="flex justify-center">
                {loading ? (
                  <Loader2 className="h-5 w-5 animate-spin text-text-secondary" />
                ) : status === "connected" ? (
                  <Badge className="bg-status-success/15 text-status-success border border-status-success/30 text-xs px-3 py-1">
                    <CheckCircle2 className="h-3 w-3 mr-1" />
                    Connected
                  </Badge>
                ) : status === "disabled" ? (
                  <Badge className="bg-yellow-500/15 text-yellow-600 border border-yellow-500/30 text-xs px-3 py-1">
                    <Pause className="h-3 w-3 mr-1" />
                    Paused
                  </Badge>
                ) : status === "pending" ? (
                  <Badge className="bg-status-pending-light text-status-pending-dark border border-status-pending/30 text-xs px-3 py-1">
                    <Loader2 className="h-3 w-3 mr-1 animate-spin" />
                    Connecting
                  </Badge>
                ) : status === "disconnected" ? (
                  <Badge variant="outline" className="border-border-color text-text-secondary text-xs px-3 py-1">
                    Not Connected
                  </Badge>
                ) : status === "error" ? (
                  <Badge className="bg-status-error/15 text-status-error border border-status-error/30 text-xs px-3 py-1">
                    <AlertCircle className="h-3 w-3 mr-1" />
                    Error
                  </Badge>
                ) : null}
              </div>
            </div>
            
            {/* Description */}
            <p className="text-sm text-text-secondary text-center line-clamp-2 min-h-[2.5rem]">{description}</p>
            
            {/* Error/Hint */}
            {error && (
              <div className="bg-status-error/10 border border-status-error/20 rounded-lg p-2 text-xs text-status-error">
                <AlertCircle className="h-3 w-3 inline mr-1" />
                {error}
              </div>
            )}
            {hint && (
              <div className="bg-blue-500/10 border border-blue-500/20 rounded-lg p-2 text-xs text-blue-600 dark:text-blue-400">
                💡 {hint}
              </div>
            )}
            
            {/* Action Buttons */}
            <div className="flex flex-col gap-2 pt-4 border-t border-border-color">
              {status === "connected" ? (
                <>
                  <div className="flex gap-2">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleIntegrationRefresh(slug)}
                      disabled={loading}
                      title="Refresh connection"
                      className="flex-1"
                    >
                      <RefreshCw className="h-4 w-4 mr-1" />
                      Refresh
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleIntegrationDisable(slug)}
                      disabled={loading}
                      title="Pause connection"
                      className="flex-1"
                    >
                      <Pause className="h-4 w-4 mr-1" />
                      Pause
                    </Button>
                  </div>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleIntegrationDisconnect(slug)}
                    disabled={loading}
                    className="w-full text-status-error hover:bg-status-error/10 hover:text-status-error"
                  >
                    {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
                    Disconnect
                  </Button>
                </>
              ) : status === "disabled" ? (
                <>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleIntegrationEnable(slug)}
                    disabled={loading}
                    title="Resume connection"
                    className="w-full"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Resume
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleIntegrationDisconnect(slug)}
                    disabled={loading}
                    className="w-full text-status-error hover:bg-status-error/10 hover:text-status-error"
                  >
                    {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
                    Disconnect
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  onClick={() => handleIntegrationConnect(slug)}
                  disabled={loading}
                  className="w-full bg-brand-teal hover:bg-brand-teal/90 text-white font-medium py-2"
                >
                  {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : null}
                  Connect
                </Button>
              )}
            </div>
          </div>
        </CardContent>
      </Card>
    )
  }



  const handleSearchChange = useCallback((query: string) => {
    setSearchQuery(query)
  }, [])

  return (
    <SidebarInset className="flex flex-col h-screen overflow-hidden">
      {/* Header with Search - Fixed height, not scrollable */}
      <div className="flex-shrink-0 sticky top-0 z-10 bg-bg-page/95 dark:bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-bg-page/60 border-b border-border-color">
        <div className="p-6">
          <div className="flex items-center justify-between gap-8">
            <div className="flex-1">
              <h1 className="text-3xl font-bold text-text-primary">Connections</h1>
              <p className="text-sm text-text-tertiary mt-1">
                Connect to {availableToolkits.length}+ external apps to extend your agent capabilities
              </p>
            </div>
            
            {/* Search Bar - Client-side component */}
            <div className="flex-shrink-0">
              <SearchBar onSearchChange={handleSearchChange} />
            </div>
          </div>
        </div>
      </div>
      
      {/* Scrollable Content Area */}
      <div className="flex-1 overflow-y-auto w-full">
        <div className="p-6">
          {toolkitsLoading ? (
            <div className="flex justify-center items-center py-24">
              <div className="text-center space-y-4">
                <Loader2 className="h-12 w-12 animate-spin text-brand-teal mx-auto" />
                <p className="text-text-secondary">Loading available integrations...</p>
              </div>
            </div>
          ) : filteredToolkits.length === 0 ? (
            <div className="text-center py-24 text-text-secondary">
              <div className="space-y-2">
                <p className="text-lg font-medium">
                  {searchQuery ? `No apps found for "${searchQuery}"` : "No integrations available"}
                </p>
                {searchQuery && (
                  <Button
                    variant="outline"
                    onClick={() => setSearchQuery("")}
                    className="mt-4"
                  >
                    Clear search
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 2xl:grid-cols-6 gap-6 pb-12">
              {filteredToolkits.map((toolkit) => renderToolkitCard(toolkit))}
            </div>
          )}
        </div>
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


