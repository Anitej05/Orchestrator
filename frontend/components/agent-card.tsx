"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Switch } from "@/components/ui/switch"
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { Star, DollarSign, MessageSquare } from "lucide-react"
import { useToast } from "@/hooks/use-toast"
import { StarRating, InteractiveStarRating } from "@/components/ui/star-rating"
import type { Agent } from "@/lib/types"

interface AgentCardProps {
  agent: Agent
  isRegistered?: boolean
  onToggleRegistration?: () => void
}

export default function AgentCard({ agent, isRegistered = false, onToggleRegistration }: AgentCardProps) {
  const [testPrompt, setTestPrompt] = useState("")
  const [isTestOpen, setIsTestOpen] = useState(false)
  const [isRatingOpen, setIsRatingOpen] = useState(false)
  const [isTesting, setIsTesting] = useState(false)
  const [currentRating, setCurrentRating] = useState(agent.rating)
  const { toast } = useToast()

  const handleRatingUpdate = (newRating: number) => {
    setCurrentRating(newRating)
    toast({
      title: "Rating submitted",
      description: `Thank you for rating ${agent.name}!`,
    })
  }

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
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6 hover:shadow-lg hover:border-blue-500 dark:hover:border-blue-400 transition-all">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="font-semibold text-lg text-gray-900 dark:text-gray-100">{agent.name}</h3>
          <div className="flex items-center space-x-2 mt-1">
            <StarRating currentRating={currentRating} readonly={true} size="sm" />
            <Badge variant={agent.status === "active" ? "default" : "secondary"} className="text-xs">
              {agent.status}
            </Badge>
          </div>
        </div>
        <div className="text-right">
          <div className="flex items-center text-green-600 dark:text-green-400">
            <DollarSign className="w-4 h-4" />
            <span className="font-semibold">{agent.price_per_call_usd}</span>
          </div>
        </div>
      </div>

      {/* Description */}
      <p className="text-gray-600 dark:text-gray-300 text-sm mb-4 line-clamp-2">{agent.description}</p>

      {/* Capabilities */}
      <div className="flex flex-wrap gap-1 mb-4">
        {agent.capabilities.slice(0, 3).map((cap, index) => (
          <Badge key={`${agent.id}-${cap}-${index}`} variant="secondary" className="text-xs">
            {cap.replace(/_/g, " ")}
          </Badge>
        ))}
        {agent.capabilities.length > 3 && (
          <Badge variant="outline" className="text-xs">
            +{agent.capabilities.length - 3}
          </Badge>
        )}
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <Dialog open={isRatingOpen} onOpenChange={setIsRatingOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <Star className="w-4 h-4 mr-1" />
              Rate
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Rate {agent.name}</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <p className="text-sm text-gray-600">
                How would you rate your experience with this agent?
              </p>
              <InteractiveStarRating
                agentId={agent.id}
                agentName={agent.name}
                currentRating={currentRating}
                onRatingUpdate={handleRatingUpdate}
              />
            </div>
          </DialogContent>
        </Dialog>

        <Dialog open={isTestOpen} onOpenChange={setIsTestOpen}>
          <DialogTrigger asChild>
            <Button variant="outline" size="sm">
              <MessageSquare className="w-4 h-4 mr-1" />
              Test
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Test {agent.name}</DialogTitle>
            </DialogHeader>
            <div className="space-y-4">
              <Textarea
                placeholder="Enter test prompt..."
                value={testPrompt}
                onChange={(e) => setTestPrompt(e.target.value)}
                rows={3}
              />
              <div className="flex justify-end space-x-2">
                <Button variant="outline" onClick={() => setIsTestOpen(false)}>
                  Cancel
                </Button>
                <Button onClick={handleTest} disabled={isTesting || !testPrompt.trim()}>
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
