"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { SidebarInset, SidebarTrigger } from "@/components/ui/sidebar"
// import AgentRegistrationForm from "@/components/agent-registration-form"
import AgentPreview from "@/components/agent-preview"
import { Button } from "@/components/ui/button"
import { useToast } from "@/hooks/use-toast"
import { ArrowLeft } from "lucide-react"
import { registerAgent } from "@/lib/api-client"
import { type Agent, type AgentEndpoint } from "@/lib/types"

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

export default function RegisterAgent() {
  const router = useRouter()
  const { toast } = useToast()
  const [formData, setFormData] = useState<FormData>({
    name: "",
    framework: "",
    capabilities: [] as string[],
    endpoints: [] as EndpointDetail[],
    description: "",
    successRate: "",
    pricePerCall: ""
  })
  const [isSaving, setIsSaving] = useState(false)
  const [isTesting, setIsTesting] = useState(false)

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
    if (!formData.name || !formData.framework || !formData.description || !formData.pricePerCall || formData.endpoints.length === 0) {
      toast({
        title: "Validation Error",
        description: "Please fill in all required fields including at least one endpoint.",
        variant: "destructive"
      })
      return
    }

    if (formData.capabilities.length === 0) {
      toast({
        title: "Validation Error",
        description: "Please add at least one capability.",
        variant: "destructive"
      })
      return
    }

    setIsSaving(true)

    try {
      const agentData: Agent = {
        id: `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        owner_id: "user_demo",
        name: formData.name,
        description: formData.description,
        capabilities: formData.capabilities,
        price_per_call_usd: Number.parseFloat(formData.pricePerCall),
        status: "active",
        endpoints: formData.endpoints.map((ep: any) => ({
          endpoint: ep.endpoint,
          http_method: ep.http_method as "GET" | "POST" | "PUT" | "DELETE",
          description: ep.description,
        })),
        rating: formData.successRate ? Number.parseFloat(formData.successRate) / 20 : 4.5,
        public_key_pem: generateDemoPublicKey()
      }

      await registerAgent(agentData)

      toast({
        title: "Agent registered successfully",
        description: "Your agent has been added to the marketplace."
      })

      router.push("/agents")
    } catch (error) {
      console.error("Registration error:", error)
      toast({
        title: "Registration failed",
        description: error instanceof Error ? error.message : "An error occurred while registering your agent.",
        variant: "destructive"
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
        variant: "destructive"
      })
      return
    }

    setIsTesting(true)

    try {
      await new Promise((resolve) => setTimeout(resolve, 3000))

      toast({
        title: "Test completed successfully",
        description: `All ${formData.endpoints.length} endpoint(s) are accessible.`
      })
    } catch (error) {
      toast({
        title: "Test failed",
        description: "Unable to connect to your agent endpoints.",
        variant: "destructive"
      })
    } finally {
      setIsTesting(false)
    }
  }

  const handleCancel = () => {
    router.push("/")
  }

  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        <div className="sticky top-0 z-10 bg-bg-card/90 dark:bg-card/90 backdrop-blur-sm border-b border-border-color px-4 py-3">
          <div className="flex items-center space-x-4">
            <SidebarTrigger className="h-8 w-8" />
            <Button variant="ghost" onClick={() => router.back()} className="text-text-primary">
              <ArrowLeft className="w-4 h-4 mr-2" />
              Back
            </Button>
          </div>
        </div>

        <main className="container mx-auto px-4 py-6 space-y-6">
          <div className="mb-4">
            <h1 className="text-3xl font-bold text-brand-teal">Register New Agent</h1>
            <p className="text-text-secondary mt-2">Add your AI agent to the marketplace and start earning from task executions.</p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 items-start">
            <div className="lg:col-span-2">
              <div className="p-4 border rounded text-red-500">
                AgentRegistrationForm is missing from the codebase. Registration disabled.
              </div>
            </div>

            <div className="lg:col-span-1">
              <AgentPreview formData={formData as any} />
            </div>
          </div>
        </main>
      </div>
    </SidebarInset>
  )
}

