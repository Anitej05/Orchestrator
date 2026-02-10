"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { Star, DollarSign, MessageSquare } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import type { Agent } from "@/lib/types"

interface AgentCardProps {
  agent: Agent
  isRegistered?: boolean
  onToggleRegistration?: () => void
}

export default function AgentCard({ agent, isRegistered = false, onToggleRegistration }: AgentCardProps) {
  const [testPrompt, setTestPrompt] = useState("")
  const [isTestOpen, setIsTestOpen] = useState(false)
  const [isTesting, setIsTesting] = useState(false)
  const { toast } = useToast()

  const handleTest = async () => {
    if (!testPrompt.trim()) return

    setIsTesting(true)
    await new Promise((resolve) => setTimeout(resolve, 2000))

    toast({
      title: "Test completed",
      description: `${agent.name} processed your request successfully.`,
    })

    setIsTestOpen(false)
    setTestPrompt("")
    setIsTesting(false)
  }

  const handleToggleRegistration = () => {
    if (onToggleRegistration) {
      onToggleRegistration()
      toast({
        title: isRegistered ? "Unregistered" : "Registered",
        description: `${agent.name} ${isRegistered ? "removed from" : "added to"} your agents.`,
      })
    }
  }

  return (
    <div className="ui-card-hover p-6">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="ui-task-name">{agent.name}</h3>
          <div className="mt-1">
            <Badge variant={agent.status === "active" ? "ui-active" : "ui-pending"}>
              {agent.status}
            </Badge>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center text-status-success-dark">
            <DollarSign className="w-4 h-4" />
            <span className="ui-metadata-mono">{agent.price_per_call_usd}</span>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="ui-task-description mb-4 line-clamp-2">{agent.description}</p>

      {/* Capabilities */}
      <div className="flex flex-wrap gap-1 mb-4">
        {agent.capabilities.slice(0, 3).map((cap, index) => (
          <Badge key={`${agent.id}-${cap}-${index}`} variant="ui-pending" className="ui-file-meta">
            {cap.replace(/_/g, " ")}
          </Badge>
        ))}
        {agent.capabilities.length > 3 && (
          <Badge variant="outline" className="ui-file-meta">
            +{agent.capabilities.length - 3}
          </Badge>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <Dialog open={isTestOpen} onOpenChange={setIsTestOpen}>
          <DialogTrigger asChild>
            <Button variant="ui-secondary" size="sm">
              <MessageSquare className="w-4 h-4 mr-1" />
              Test
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle className="ui-section-header">Test {agent.name}</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <Textarea
                className="ui-textarea"
                placeholder="Enter test prompt..."
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                rows={3}
              />
              <div className="flex justify-end space-x-2">
                <Button variant="ui-secondary" onClick={() => setIsTestOpen(false)}>
                  Cancel
                </Button>
                <Button variant="ui-primary" onClick={handleTest} disabled={isTesting || !testPrompt.trim()}>
                  {isTesting ? "Testing..." : "Test"}
                </Button>
              </div>
            </div>
          </DialogContent>
        </Dialog>
      </div>
    </div>
  )
}

