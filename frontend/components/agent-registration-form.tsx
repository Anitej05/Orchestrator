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
    <div className="ui-card p-6">
      <h2 className="ui-section-header text-brand-teal mb-6">Agent Details</h2>

      <div className="space-y-6">
        {/* Basic Information */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="ui-metadata-label block mb-2">Agent Name *</label>
            <Input
              className="ui-input"
              placeholder="My Awesome Agent"
              value={formData.name}
              onChange={(e) => updateField("name", e.target.value)}
            />
          </div>

          <div>
            <label className="ui-metadata-label block mb-2">Framework *</label>
            <Select value={formData.framework} onValueChange={(value) => updateField("framework", value)}>
              <SelectTrigger className="ui-input">
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
          <label className="ui-metadata-label block mb-2">Capability Tags *</label>
          <Select onValueChange={addCapability}>
            <SelectTrigger className="ui-input">
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
              <Badge key={capability} variant="ui-pending" className="flex items-center gap-1">
                {capability}
                <X className="w-3 h-3 cursor-pointer" onClick={() => removeCapability(capability)} />
              </Badge>
            ))}
          </div>
        </div>

        {/* Endpoints */}
        <div>
          <label className="ui-metadata-label block mb-2">Endpoints *</label>

          {/* Existing Endpoints */}
          <div className="space-y-3 mb-4">
            {formData.endpoints.map((endpoint, index) => (
              <div key={`${endpoint.endpoint}-${index}`} className="ui-metadata-item p-4">
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                  <div>
                    <label className="ui-file-meta block mb-1">URL</label>
                    <Input
                      className="ui-input"
                      value={endpoint.endpoint}
                      onChange={(e) => updateEndpoint(index, "endpoint", e.target.value)}
                      placeholder="https://api.example.com/endpoint"
                    />
                  </div>
                  <div>
                    <label className="ui-file-meta block mb-1">Method</label>
                    <Select
                      value={endpoint.http_method}
                      onValueChange={(value) => updateEndpoint(index, "http_method", value)}
                    >
                      <SelectTrigger className="ui-input">
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
                      variant="ui-secondary"
                      size="sm"
                      onClick={() => removeEndpoint(index)}
                      className="w-full"
                    >
                      <X className="w-4 h-4" />
                    </Button>
                  </div>
                </div>
                <div className="mt-3">
                  <label className="ui-file-meta block mb-1">Description</label>
                  <Input
                    className="ui-input"
                    value={endpoint.description || ""}
                    onChange={(e) => updateEndpoint(index, "description", e.target.value)}
                    placeholder="Endpoint description"
                  />
                </div>
              </div>
            ))}
          </div>

          {/* Add New Endpoint */}
          <div className="border-2 border-brand-teal/20 rounded-orbimesh-lg p-4 bg-brand-teal-light/50">
            <h4 className="ui-metadata-label text-brand-teal mb-3">Add New Endpoint</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div>
                <Input
                  className="ui-input"
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
                  <SelectTrigger className="ui-input">
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
                <Button type="button" variant="ui-primary" onClick={addEndpoint} size="sm" className="w-full">
                  <Plus className="w-4 h-4 mr-1" />
                  Add
                </Button>
              </div>
            </div>
            <div className="mt-3">
              <Input
                className="ui-input"
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
            <label className="ui-metadata-label block mb-2">Price per Call (USD) *</label>
            <Input
              className="ui-input"
              type="number"
              step="0.01"
              placeholder="1.50"
              value={formData.pricePerCall}
              onChange={(e) => updateField("pricePerCall", e.target.value)}
            />
          </div>

          <div>
            <label className="ui-metadata-label block mb-2">Success Rate (%)</label>
            <Input
              className="ui-input"
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
          <label className="ui-metadata-label block mb-2">Agent Description *</label>
          <Textarea
            className="ui-textarea"
            placeholder="Describe what your agent does, its strengths, and ideal use cases..."
            rows={4}
            value={formData.description}
            onChange={(e) => updateField("description", e.target.value)}
          />
        </div>

        {/* Action Buttons */}
        <div className="flex flex-col sm:flex-row gap-3 pt-6 border-t border-border-color">
          <Button variant="ui-primary" onClick={onSave} className="flex-1" disabled={isSaving || isTesting}>
            {isSaving ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                Saving...
              </>
            ) : (
              "Save Agent"
            )}
          </Button>
          <Button onClick={onTest} variant="ui-secondary" className="flex-1" disabled={isSaving || isTesting}>
            {isTesting ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-text-tertiary mr-2"></div>
                Testing...
              </>
            ) : (
              "Test Agent"
            )}
          </Button>
          <Button variant="ghost" className="flex-1 ui-nav-link" onClick={onCancel} disabled={isSaving || isTesting}>
            Cancel
          </Button>
        </div>
      </div>
    </div>
  )
}

