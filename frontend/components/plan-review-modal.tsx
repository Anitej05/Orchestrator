'use client'

import { useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { CheckCircle2, AlertCircle, Edit2 } from 'lucide-react'
import type { TaskAgentPair } from '@/lib/types'

interface PlanReviewModalProps {
  isOpen: boolean
  plan: any[]
  taskAgentPairs: TaskAgentPair[]
  originalPrompt: string
  onAccept: (modifiedPrompt?: string) => void
  onReject: () => void
  isLoading?: boolean
}

export function PlanReviewModal({
  isOpen,
  plan,
  taskAgentPairs,
  originalPrompt,
  onAccept,
  onReject,
  isLoading = false
}: PlanReviewModalProps) {
  const [modifyMode, setModifyMode] = useState(false)
  const [modifiedPrompt, setModifiedPrompt] = useState(originalPrompt)

  if (!isOpen) return null

  const handleAccept = () => {
    onAccept(modifyMode ? modifiedPrompt : undefined)
    setModifyMode(false)
    setModifiedPrompt(originalPrompt)
  }

  const handleReject = () => {
    onReject()
    setModifyMode(false)
    setModifiedPrompt(originalPrompt)
  }

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <CheckCircle2 className="w-5 h-5 text-green-600" />
            Review Workflow Plan
          </CardTitle>
          <CardDescription>
            Review the execution plan before running the workflow
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Original Prompt */}
          <div className="space-y-2">
            <h3 className="font-semibold text-sm">Original Request</h3>
            <div className="bg-gray-100 dark:bg-gray-800 p-3 rounded-md text-sm">
              {originalPrompt}
            </div>
          </div>

          {/* Task Breakdown */}
          <div className="space-y-2">
            <h3 className="font-semibold text-sm">Execution Plan ({taskAgentPairs.length} tasks)</h3>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {taskAgentPairs.map((pair, idx) => (
                <div
                  key={idx}
                  className="border border-gray-200 dark:border-gray-700 rounded-md p-3 text-sm"
                >
                  <div className="flex items-start gap-2">
                    <div className="font-semibold text-blue-600 min-w-fit">Task {idx + 1}</div>
                    <div className="flex-1">
                      <p className="font-medium">{pair.task_name || pair.task_description}</p>
                      <p className="text-gray-600 dark:text-gray-400 text-xs mt-1">
                        Agent: <span className="font-semibold">{pair.primary?.name || 'Unknown'}</span>
                      </p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Modify Prompt Option */}
          {!modifyMode ? (
            <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md p-3 flex items-start gap-3">
              <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
              <div className="flex-1">
                <p className="text-sm font-medium text-blue-900 dark:text-blue-200">
                  Want to modify the plan?
                </p>
                <p className="text-sm text-blue-800 dark:text-blue-300 mt-1">
                  You can change the prompt to adjust which agents are used or how tasks are executed.
                </p>
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2 gap-2"
                  onClick={() => setModifyMode(true)}
                >
                  <Edit2 className="w-4 h-4" />
                  Modify Prompt
                </Button>
              </div>
            </div>
          ) : (
            <div className="space-y-2">
              <label className="text-sm font-medium">Modified Prompt</label>
              <Textarea
                value={modifiedPrompt}
                onChange={(e) => setModifiedPrompt(e.target.value)}
                placeholder="Enter your modified request..."
                rows={4}
              />
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setModifyMode(false)
                    setModifiedPrompt(originalPrompt)
                  }}
                >
                  Cancel
                </Button>
                <Button size="sm" variant="secondary">
                  Save & Re-plan (creates new plan)
                </Button>
              </div>
            </div>
          )}

          {/* Action Buttons */}
          <div className="flex gap-2 pt-4 border-t">
            <Button
              variant="outline"
              onClick={handleReject}
              disabled={isLoading}
              className="flex-1"
            >
              Cancel
            </Button>
            <Button
              onClick={handleAccept}
              disabled={isLoading}
              className="flex-1 gap-2"
            >
              {isLoading ? 'Starting...' : 'Accept & Execute'}
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
