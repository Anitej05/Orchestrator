"use client"

import { useState } from "react"
import { Star, Loader2 } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface StarRatingProps {
  currentRating?: number
  onRate?: (rating: number) => void
  readonly?: boolean
  size?: "sm" | "md" | "lg"
  className?: string
  showValue?: boolean
}

export function StarRating({
  currentRating = 0,
  onRate,
  readonly = false,
  size = "md",
  className,
  showValue = true
}: StarRatingProps) {
  const [hoveredRating, setHoveredRating] = useState<number | null>(null)

  const sizes = {
    sm: "w-3.0 h-3.0", // Reduced the size of the 'sm' stars
    md: "w-5 h-5",
    lg: "w-6 h-6"
  }

  const displayRating = hoveredRating ?? currentRating

  const handleStarClick = (rating: number) => {
    if (!readonly && onRate) {
      onRate(rating)
    }
  }

  const handleMouseEnter = (rating: number) => {
    if (!readonly) {
      setHoveredRating(rating)
    }
  }

  const handleMouseLeave = () => {
    if (!readonly) {
      setHoveredRating(null)
    }
  }

  return (
    <div className={cn("flex items-center", className)}>
      <div className="flex items-center space-x-0"> {/* Removed all spacing between stars */}
        {[1, 2, 3, 4, 5].map((star) => (
          <Button
            key={star}
            variant="ghost"
            size="icon" // Use icon size for a smaller footprint
            className={cn(
              "p-0 h-auto w-auto hover:bg-transparent",
              !readonly && "cursor-pointer",
              readonly && "cursor-default"
            )}
            onClick={() => handleStarClick(star)}
            onMouseEnter={() => handleMouseEnter(star)}
            onMouseLeave={handleMouseLeave}
            disabled={readonly}
          >
            <Star
              className={cn(
                sizes[size],
                "transition-colors duration-150",
                displayRating >= star
                  ? "text-yellow-400 fill-yellow-400"
                  : "text-gray-300"
              )}
            />
          </Button>
        ))}
      </div>
      {showValue && (
        <div className="flex items-center space-x-1 ml-2">
          <span className="text-sm text-gray-600 font-medium tabular-nums">
            ({(currentRating ?? 0).toFixed(1)})
          </span>
          <Star className="w-3.5 h-3.5 text-yellow-400 fill-yellow-400" />
        </div>
      )}
    </div>
  )
}

interface InteractiveStarRatingProps {
  agentId: string
  agentName: string
  currentRating?: number
  onRatingUpdate?: (newRating: number) => void
  className?: string
}

export function InteractiveStarRating({
  agentId,
  agentName,
  currentRating,
  onRatingUpdate,
  className
}: InteractiveStarRatingProps) {
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [userRating, setUserRating] = useState<number | null>(null)
  const [message, setMessage] = useState<string>("")

  const handleRate = async (rating: number) => {
    setIsSubmitting(true)
    setMessage("")

    try {
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'

      const response = await fetch(`${API_BASE_URL}/api/agents/${encodeURIComponent(agentId)}/rate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ rating }),
      })

      if (!response.ok) {
        const nameResponse = await fetch(`${API_BASE_URL}/api/agents/by-name/${encodeURIComponent(agentName)}/rate`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ rating }),
        })

        if (!nameResponse.ok) {
          throw new Error(`Failed to rate agent: ${nameResponse.status}`)
        }

        const updatedAgent = await nameResponse.json()
        onRatingUpdate?.(updatedAgent.rating)
      } else {
        const updatedAgent = await response.json()
        onRatingUpdate?.(updatedAgent.rating)
      }

      setUserRating(rating)
      setMessage("Thank you!")
      setTimeout(() => setMessage(""), 3000)

    } catch (error) {
      console.error('Error rating agent:', error)
      setMessage("Failed.")
      setTimeout(() => setMessage(""), 3000)
    } finally {
      setIsSubmitting(false)
    }
  }

  return (
    <div className={cn("flex items-center justify-end gap-x-2", className)}>
        {/* Interactive stars for user input */}
        <StarRating
          currentRating={userRating ?? currentRating ?? 0} // Show user's selection, fallback to currentRating, then to 0
          onRate={handleRate}
          readonly={isSubmitting}
          showValue={false} // Hides the (4.2) ★ part
          size="sm"
        />

        {/* Status Messages */}
        <div className="w-20 h-4 text-left">
            {isSubmitting && <Loader2 className="w-4 h-4 animate-spin text-gray-400" />}
            {message && (
                <span className={cn(
                "text-xs",
                message.includes("Thank you") ? "text-green-600" : "text-red-600"
                )}>
                {message}
                </span>
            )}
        </div>
    </div>
  )
}
