"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import AppSidebar from "@/components/app-sidebar"
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
import AgentRegistrationForm from "@/components/agent-registration-form"
import AgentPreview from "@/components/agent-preview"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { ArrowLeft } from "lucide-react"
import { registerAgent } from "@/lib/api-client"

interface EndpointDetail {
  endpoint: string
  http_method: string
  description?: string
}

interface AgentRegistrationData {
  id: string
  owner_id: string
  name: string
  description: string
  capabilities: string[]
  price_per_call_usd: number
  status: "active" | "inactive" | "deprecated"
  endpoints: EndpointDetail[]
  rating: number
  public_key_pem: string
}

export default function RegisterAgent() {
  const router = useRouter()
  const { toast } = useToast()
  const [formData, setFormData] = useState({
    name: "",
    framework: "",
    capabilities: [] as string[],
    endpoints: [] as EndpointDetail[],
    description: "",
    successRate: "",
    pricePerCall: "",
  })
  const [isSaving, setIsSaving] = useState(false)
  const [isTesting, setIsTesting] = useState(false)

  // Generate a simple public key for demo purposes
  const generateDemoPublicKey = () => {
    return `-----BEGIN PUBLIC KEY-----
MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8AMIIBCgKCAQEA1234567890abcdefghij
klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij
klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij
klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij
klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890abcdefghij
klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ
-----END PUBLIC KEY-----`
  }

  const handleSave = async () => {
    // Validate required fields
    if (
      !formData.name ||
      !formData.framework ||
      !formData.description ||
      !formData.pricePerCall ||
      formData.endpoints.length === 0
    ) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields including at least one endpoint.",
        variant: "destructive",
      })
      return
    }

    if (formData.capabilities.length === 0) {
      toast({
        title: "Validation Error",
        description: "Please add at least one capability.",
        variant: "destructive",
      })
      return
    }

    setIsSaving(true)

    try {
      const agentData: AgentRegistrationData = {
        id: `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        owner_id: "user_demo", // In a real app, this would come from authentication
        name: formData.name,
        description: formData.description,
        capabilities: formData.capabilities,
        price_per_call_usd: Number.parseFloat(formData.pricePerCall),
        status: "active",
        endpoints: formData.endpoints,
        rating: formData.successRate ? Number.parseFloat(formData.successRate) / 20 : 4.5, // Convert percentage to 5-star rating
        public_key_pem: generateDemoPublicKey(),
      }

      await registerAgent(agentData)

      toast({
        title: "Agent registered successfully",
        description: "Your agent has been added to the marketplace.",
      })

      router.push("/agents")
    } catch (error) {
      console.error("Registration error:", error)
      toast({
        title: "Registration failed",
        description: error instanceof Error ? error.message : "An error occurred while registering your agent.",
        variant: "destructive",
      })
    } finally {
      setIsSaving(false)
    }
  }

  const handleTest = async () => {
    if (formData.endpoints.length === 0) {
      toast({
        title: "Test Error",
        description: "Please add at least one endpoint to test the agent.",
        variant: "destructive",
      })
      return
    }

    setIsTesting(true)

    try {
      // Simulate testing agent endpoints
      await new Promise((resolve) => setTimeout(resolve, 3000))

      toast({
        title: "Test completed successfully",
        description: `All ${formData.endpoints.length} endpoint(s) are accessible.`,
      })
    } catch (error) {
      toast({
        title: "Test failed",
        description: "Unable to connect to your agent endpoints.",
        variant: "destructive",
      })
    } finally {
      setIsTesting(false)
    }
  }

  const handleCancel = () => {
    router.push("/")
  }

  return (
    <SidebarProvider>
      <AppSidebar />
      <SidebarInset>
        <div className="min-h-screen bg-gray-50">
          {/* Header */}
          <div className="sticky top-0 z-10 bg-white border-b border-gray-200 px-4 py-3">
            <div className="flex items-center space-x-4">
              <SidebarTrigger className="h-8 w-8" />
              <Button variant="ghost" onClick={() => router.back()}>
                <ArrowLeft className="w-4 h-4 mr-2" />
                Back
              </Button>
            </div>
          </div>

          <main className="container mx-auto px-4 py-6">
            <div className="mb-6">
              <h1 className="text-3xl font-bold text-gray-900">Register New Agent</h1>
              <p className="text-gray-600 mt-2">
                Add your AI agent to the marketplace and start earning from task executions.
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Registration Form - 2/3 width */}
              <div className="lg:col-span-2">
                <AgentRegistrationForm
                  formData={formData}
                  setFormData={setFormData}
                  onSave={handleSave}
                  onTest={handleTest}
                  onCancel={handleCancel}
                  isSaving={isSaving}
                  isTesting={isTesting}
                />
              </div>

              {/* Live Preview - 1/3 width */}
              <div className="lg:col-span-1">
                <AgentPreview formData={formData} />
              </div>
            </div>
          </main>
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}
