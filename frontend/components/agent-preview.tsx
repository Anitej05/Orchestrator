import { Badge } from "@/components/ui/badge"
import { Star, DollarSign, Zap, ExternalLink } from "lucide-react"
import { StarRating } from "@/components/ui/star-rating"
import type { AgentEndpoint } from "@/lib/types"

interface FormData {
  name: string
  framework: string
  capabilities: string[]
  endpoints: AgentEndpoint[]
  description: string
  successRate: string
  pricePerCall: string
}

interface AgentPreviewProps {
  formData: FormData
}

export default function AgentPreview({ formData }: AgentPreviewProps) {
  const displayName = formData.name || "Agent Name"
  const displayFramework = formData.framework || "Framework"
  const displayPrice = formData.pricePerCall || "0.00"
  const displaySuccessRate = formData.successRate || "0"
  const displayDescription = formData.description || "Agent description will appear here..."
  const displayRating = formData.successRate ? (Number.parseFloat(formData.successRate) / 20).toFixed(1) : "0.0"

  return (
    <div className="bg-white rounded-lg shadow-sm p-6 sticky top-24">
      <h2 className="text-xl font-bold text-gray-900 mb-6">Live Preview</h2>

      {/* Preview Card */}
      <div className="bg-white rounded-lg border border-gray-200 p-6 hover:shadow-lg transition-shadow duration-200">
        {/* Header */}
        <div className="flex items-start justify-between mb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold">
              {displayName.charAt(0)}
            </div>
            <div>
              <h3 className="font-semibold text-gray-900">{displayName}</h3>
              <p className="text-sm text-gray-500">{displayFramework}</p>
            </div>
          </div>
        </div>

        {/* Status Badge */}
        <div className="mb-4">
          <Badge className="bg-green-100 text-green-800">ACTIVE</Badge>
        </div>

        {/* Capabilities */}
        <div className="flex flex-wrap gap-2 mb-4">
          {formData.capabilities.length > 0 ? (
            formData.capabilities.map((capability) => (
              <Badge key={capability} variant="secondary" className="text-xs">
                {capability.replace(/_/g, " ")}
              </Badge>
            ))
          ) : (
            <Badge variant="secondary" className="text-xs opacity-50">
              No capabilities selected
            </Badge>
          )}
        </div>

        {/* Metrics */}
        <div className="flex items-center justify-between mb-4">
          <StarRating currentRating={Number.parseFloat(displayRating)} readonly={true} size="sm" />
          <div className="flex items-center space-x-1">
            <DollarSign className="w-4 h-4 text-blue-500" />
            <span className="text-sm font-medium">${displayPrice}</span>
          </div>
          <div className="flex items-center space-x-1">
            <Zap className="w-4 h-4 text-green-500" />
            <span className="text-sm font-medium">
              {formData.endpoints.length} endpoint{formData.endpoints.length !== 1 ? "s" : ""}
            </span>
          </div>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-600 mb-4 line-clamp-3">{displayDescription}</p>

        {/* Endpoints Preview */}
        {formData.endpoints.length > 0 && (
          <div className="mb-4">
            <h4 className="text-xs font-medium text-gray-700 mb-2">Endpoints:</h4>
            <div className="space-y-1">
              {formData.endpoints.slice(0, 2).map((endpoint, index) => (
                <div key={`${endpoint.endpoint}-${index}`} className="flex items-center text-xs text-gray-500">
                  <Badge variant="outline" className="mr-2 text-xs">
                    {endpoint.http_method}
                  </Badge>
                  <span className="truncate flex-1">{endpoint.endpoint}</span>
                  <ExternalLink className="w-3 h-3 ml-1" />
                </div>
              ))}
              {formData.endpoints.length > 2 && (
                <div className="text-xs text-gray-400">
                  +{formData.endpoints.length - 2} more endpoint{formData.endpoints.length - 2 !== 1 ? "s" : ""}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Preview Note */}
        <div className="text-xs text-gray-400 italic">This is how your agent will appear in the marketplace</div>
      </div>
    </div>
  )
}
