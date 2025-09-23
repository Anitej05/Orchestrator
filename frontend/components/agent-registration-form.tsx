"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { X, Plus } from "lucide-react"
import { capabilities, frameworks } from "@/lib/api-client"

interface EndpointDetail {
  endpoint: string
  http_method: string
  description?: string
}

interface FormData {
  name: string
  framework: string
  capabilities: string[]
  endpoints: EndpointDetail[]
  description: string
  successRate: string
  pricePerCall: string
}

interface AgentRegistrationFormProps {
  formData: FormData
  setFormData: (data: FormData) => void
  onSave: () => void
  onTest: () => void
  onCancel?: () => void
  isSaving?: boolean
  isTesting?: boolean
}

export default function AgentRegistrationForm({
  formData,
  setFormData,
  onSave,
  onTest,
  onCancel,
  isSaving = false,
  isTesting = false,
}: AgentRegistrationFormProps) {
  const [newEndpoint, setNewEndpoint] = useState<EndpointDetail>({
    endpoint: "",
    http_method: "POST",
    description: "",
  })

  const updateField = (field: keyof FormData, value: string | string[] | EndpointDetail[]) => {
    setFormData({ ...formData, [field]: value })
  }

  const addCapability = (capability: string) => {
    if (!formData.capabilities.includes(capability)) {
      updateField("capabilities", [...formData.capabilities, capability])
    }
  }

  const removeCapability = (capability: string) => {
    updateField(
      "capabilities",
      formData.capabilities.filter((c) => c !== capability),
    )
  }

  const addEndpoint = () => {
    if (newEndpoint.endpoint.trim()) {
      updateField("endpoints", [...formData.endpoints, { ...newEndpoint }])
      setNewEndpoint({
        endpoint: "",
        http_method: "POST",
        description: "",
      })
    }
  }

  const removeEndpoint = (index: number) => {
    updateField(
      "endpoints",
      formData.endpoints.filter((_, i) => i !== index),
    )
  }

  const updateEndpoint = (index: number, field: keyof EndpointDetail, value: string) => {
    const updatedEndpoints = formData.endpoints.map((endpoint, i) =>
      i === index ? { ...endpoint, [field]: value } : endpoint,
    )
    updateField("endpoints", updatedEndpoints)
  }

  return (
    <div className="bg-white rounded-lg shadow-sm p-6">
      <h2 className="text-xl font-bold text-gray-900 mb-6">Agent Details</h2>

      <div className="space-y-6">
        {/* Basic Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Agent Name *</label>
            <Input
              placeholder="My Awesome Agent"
              value={formData.name}
              onChange={(e) => updateField("name", e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Framework *</label>
            <Select value={formData.framework} onValueChange={(value) => updateField("framework", value)}>
              <SelectTrigger>
                <SelectValue placeholder="Select framework" />
              </SelectTrigger>
              <SelectContent>
                {frameworks.map((framework) => (
                  <SelectItem key={framework} value={framework}>
                    {framework}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>

        {/* Capabilities */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Capability Tags *</label>
          <Select onValueChange={addCapability}>
            <SelectTrigger>
              <SelectValue placeholder="Add capabilities" />
            </SelectTrigger>
            <SelectContent>
              {capabilities
                .filter((cap) => !formData.capabilities.includes(cap))
                .map((capability) => (
                  <SelectItem key={capability} value={capability}>
                    {capability}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
          <div className="flex flex-wrap gap-2 mt-2">
            {formData.capabilities.map((capability) => (
              <Badge key={capability} variant="secondary" className="flex items-center gap-1">
                {capability}
                <X className="w-3 h-3 cursor-pointer" onClick={() => removeCapability(capability)} />
              </Badge>
            ))}
          </div>
        </div>

        {/* Endpoints */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Endpoints *</label>

          {/* Existing Endpoints */}
          <div className="space-y-3 mb-4">
            {formData.endpoints.map((endpoint, index) => (
              <div key={index} className="border rounded-lg p-4 bg-gray-50">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">URL</label>
                    <Input
                      value={endpoint.endpoint}
                      onChange={(e) => updateEndpoint(index, "endpoint", e.target.value)}
                      placeholder="https://api.example.com/endpoint"
                    />
                  </div>
                  <div>
                    <label className="block text-xs font-medium text-gray-600 mb-1">Method</label>
                    <Select
                      value={endpoint.http_method}
                      onValueChange={(value) => updateEndpoint(index, "http_method", value)}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="GET">GET</SelectItem>
                        <SelectItem value="POST">POST</SelectItem>
                        <SelectItem value="PUT">PUT</SelectItem>
                        <SelectItem value="DELETE">DELETE</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="flex items-end">
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => removeEndpoint(index)}
                      className="w-full"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                <div className="mt-3">
                  <label className="block text-xs font-medium text-gray-600 mb-1">Description</label>
                  <Input
                    value={endpoint.description || ""}
                    onChange={(e) => updateEndpoint(index, "description", e.target.value)}
                    placeholder="Endpoint description"
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Add New Endpoint */}
          <div className="border rounded-lg p-4 bg-blue-50">
            <h4 className="text-sm font-medium text-gray-700 mb-3">Add New Endpoint</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <Input
                  value={newEndpoint.endpoint}
                  onChange={(e) => setNewEndpoint({ ...newEndpoint, endpoint: e.target.value })}
                  placeholder="https://api.example.com/endpoint"
                />
              </div>
              <div>
                <Select
                  value={newEndpoint.http_method}
                  onValueChange={(value) => setNewEndpoint({ ...newEndpoint, http_method: value })}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="GET">GET</SelectItem>
                    <SelectItem value="POST">POST</SelectItem>
                    <SelectItem value="PUT">PUT</SelectItem>
                    <SelectItem value="DELETE">DELETE</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-end">
                <Button type="button" onClick={addEndpoint} size="sm" className="w-full">
                  <Plus className="w-4 h-4 mr-1" />
                  Add
                </Button>
              </div>
            </div>
            <div className="mt-3">
              <Input
                value={newEndpoint.description || ""}
                onChange={(e) => setNewEndpoint({ ...newEndpoint, description: e.target.value })}
                placeholder="Endpoint description"
              />
            </div>
          </div>
        </div>

        {/* Pricing and Performance */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Price per Call (USD) *</label>
            <Input
              type="number"
              step="0.01"
              placeholder="1.50"
              value={formData.pricePerCall}
              onChange={(e) => updateField("pricePerCall", e.target.value)}
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Success Rate (%)</label>
            <Input
              type="number"
              min="0"
              max="100"
              placeholder="95"
              value={formData.successRate}
              onChange={(e) => updateField("successRate", e.target.value)}
            />
          </div>
        </div>

        {/* Description */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">Agent Description *</label>
          <Textarea
            placeholder="Describe what your agent does, its strengths, and ideal use cases..."
            rows={4}
            value={formData.description}
            onChange={(e) => updateField("description", e.target.value)}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 pt-6 border-t">
          <Button onClick={onSave} className="flex-1" disabled={isSaving || isTesting}>
            {isSaving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Saving...
              </>
            ) : (
              "Save Agent"
            )}
          </Button>
          <Button onClick={onTest} variant="outline" className="flex-1 bg-transparent" disabled={isSaving || isTesting}>
            {isTesting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-gray-600 mr-2"></div>
                Testing...
              </>
            ) : (
              "Test Agent"
            )}
          </Button>
          <Button variant="ghost" className="flex-1" onClick={onCancel} disabled={isSaving || isTesting}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  )
}
