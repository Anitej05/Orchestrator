"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@clerk/nextjs";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { 
  CheckCircle2, 
  XCircle, 
  Key, 
  Eye, 
  EyeOff, 
  Save, 
  Trash2, 
  AlertCircle,
  Shield,
  Loader2
} from "lucide-react";

interface CredentialField {
  name: string;
  label: string;
  type: string;
  required: boolean;
  description: string;
  placeholder?: string;
}

interface AgentCredentialStatus {
  agent_id: string;
  agent_name: string;
  agent_type: string;
  requires_credentials: boolean;
  credential_fields: CredentialField[];
  is_configured: boolean;
  configured_fields: string[];
  created_at?: string;
  updated_at?: string;
}

export default function CredentialsPage() {
  const { getToken } = useAuth();
  const [agents, setAgents] = useState<AgentCredentialStatus[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState<string | null>(null);
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [credentials, setCredentials] = useState<Record<string, string>>({});
  const [showValues, setShowValues] = useState<Record<string, boolean>>({});
  const [message, setMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  useEffect(() => {
    loadCredentialsStatus();
  }, []);

  const loadCredentialsStatus = async () => {
    try {
      const token = await getToken();
      const response = await fetch("http://localhost:8000/api/credentials/status", {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to load credentials");

      const data = await response.json();
      setAgents(data.agents || []);
    } catch (error) {
      console.error("Error loading credentials:", error);
      setMessage({ type: "error", text: "Failed to load credentials status" });
    } finally {
      setLoading(false);
    }
  };

  const handleSaveCredentials = async (agentId: string) => {
    setSaving(agentId);
    setMessage(null);

    try {
      const token = await getToken();
      const agent = agents.find((a) => a.agent_id === agentId);
      if (!agent) return;

      // Prepare credentials array
      const credentialsArray = agent.credential_fields.map((field) => ({
        field_name: field.name,
        value: credentials[field.name] || "",
      }));

      const response = await fetch(`http://localhost:8000/api/credentials/${agentId}`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          agent_id: agentId,
          credentials: credentialsArray,
        }),
      });

      if (!response.ok) throw new Error("Failed to save credentials");

      setMessage({ type: "success", text: "Credentials saved successfully!" });
      
      // Reload status
      await loadCredentialsStatus();
      
      // Clear form
      setCredentials({});
      setSelectedAgent(null);
    } catch (error) {
      console.error("Error saving credentials:", error);
      setMessage({ type: "error", text: "Failed to save credentials" });
    } finally {
      setSaving(null);
    }
  };

  const handleDeleteCredentials = async (agentId: string) => {
    if (!confirm("Are you sure you want to delete these credentials?")) return;

    try {
      const token = await getToken();
      const response = await fetch(`http://localhost:8000/api/credentials/${agentId}`, {
        method: "DELETE",
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to delete credentials");

      setMessage({ type: "success", text: "Credentials deleted successfully!" });
      await loadCredentialsStatus();
    } catch (error) {
      console.error("Error deleting credentials:", error);
      setMessage({ type: "error", text: "Failed to delete credentials" });
    }
  };

  const toggleShowValue = (fieldName: string) => {
    setShowValues((prev) => ({ ...prev, [fieldName]: !prev[fieldName] }));
  };

  const agentsNeedingCredentials = agents.filter((a) => a.requires_credentials);
  const configuredAgents = agentsNeedingCredentials.filter((a) => a.is_configured);
  const unconfiguredAgents = agentsNeedingCredentials.filter((a) => !a.is_configured);

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <Loader2 className="h-8 w-8 animate-spin" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4 max-w-6xl">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">Agent Credentials</h1>
        <p className="text-muted-foreground">
          Manage API keys and credentials for your agents
        </p>
      </div>

      {message && (
        <Alert className={`mb-6 ${message.type === "error" ? "border-red-500" : "border-green-500"}`}>
          <AlertCircle className="h-4 w-4" />
          <AlertDescription>{message.text}</AlertDescription>
        </Alert>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Total Agents</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{agents.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Configured</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{configuredAgents.length}</div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm font-medium">Need Setup</CardTitle>
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">{unconfiguredAgents.length}</div>
          </CardContent>
        </Card>
      </div>

      {/* Tabs */}
      <Tabs defaultValue="all" className="space-y-4">
        <TabsList>
          <TabsTrigger value="all">All Agents ({agentsNeedingCredentials.length})</TabsTrigger>
          <TabsTrigger value="configured">Configured ({configuredAgents.length})</TabsTrigger>
          <TabsTrigger value="unconfigured">Need Setup ({unconfiguredAgents.length})</TabsTrigger>
        </TabsList>

        <TabsContent value="all" className="space-y-4">
          {agentsNeedingCredentials.map((agent) => (
            <AgentCredentialCard
              key={agent.agent_id}
              agent={agent}
              isExpanded={selectedAgent === agent.agent_id}
              onToggle={() => setSelectedAgent(selectedAgent === agent.agent_id ? null : agent.agent_id)}
              credentials={credentials}
              onCredentialChange={(name, value) => setCredentials((prev) => ({ ...prev, [name]: value }))}
              showValues={showValues}
              onToggleShowValue={toggleShowValue}
              onSave={() => handleSaveCredentials(agent.agent_id)}
              onDelete={() => handleDeleteCredentials(agent.agent_id)}
              isSaving={saving === agent.agent_id}
            />
          ))}
        </TabsContent>

        <TabsContent value="configured" className="space-y-4">
          {configuredAgents.map((agent) => (
            <AgentCredentialCard
              key={agent.agent_id}
              agent={agent}
              isExpanded={selectedAgent === agent.agent_id}
              onToggle={() => setSelectedAgent(selectedAgent === agent.agent_id ? null : agent.agent_id)}
              credentials={credentials}
              onCredentialChange={(name, value) => setCredentials((prev) => ({ ...prev, [name]: value }))}
              showValues={showValues}
              onToggleShowValue={toggleShowValue}
              onSave={() => handleSaveCredentials(agent.agent_id)}
              onDelete={() => handleDeleteCredentials(agent.agent_id)}
              isSaving={saving === agent.agent_id}
            />
          ))}
        </TabsContent>

        <TabsContent value="unconfigured" className="space-y-4">
          {unconfiguredAgents.map((agent) => (
            <AgentCredentialCard
              key={agent.agent_id}
              agent={agent}
              isExpanded={selectedAgent === agent.agent_id}
              onToggle={() => setSelectedAgent(selectedAgent === agent.agent_id ? null : agent.agent_id)}
              credentials={credentials}
              onCredentialChange={(name, value) => setCredentials((prev) => ({ ...prev, [name]: value }))}
              showValues={showValues}
              onToggleShowValue={toggleShowValue}
              onSave={() => handleSaveCredentials(agent.agent_id)}
              onDelete={() => handleDeleteCredentials(agent.agent_id)}
              isSaving={saving === agent.agent_id}
            />
          ))}
        </TabsContent>
      </Tabs>
    </div>
  );
}

// Agent Credential Card Component
function AgentCredentialCard({
  agent,
  isExpanded,
  onToggle,
  credentials,
  onCredentialChange,
  showValues,
  onToggleShowValue,
  onSave,
  onDelete,
  isSaving,
}: {
  agent: AgentCredentialStatus;
  isExpanded: boolean;
  onToggle: () => void;
  credentials: Record<string, string>;
  onCredentialChange: (name: string, value: string) => void;
  showValues: Record<string, boolean>;
  onToggleShowValue: (name: string) => void;
  onSave: () => void;
  onDelete: () => void;
  isSaving: boolean;
}) {
  return (
    <Card>
      <CardHeader className="cursor-pointer" onClick={onToggle}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Shield className="h-5 w-5 text-muted-foreground" />
            <div>
              <CardTitle className="text-lg">{agent.agent_name}</CardTitle>
              <CardDescription className="text-sm">
                {agent.agent_type === "mcp_http" ? "MCP Agent" : "REST Agent"}
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {agent.is_configured ? (
              <Badge variant="default" className="bg-green-600">
                <CheckCircle2 className="h-3 w-3 mr-1" />
                Configured
              </Badge>
            ) : (
              <Badge variant="secondary" className="bg-orange-600 text-white">
                <AlertCircle className="h-3 w-3 mr-1" />
                Setup Required
              </Badge>
            )}
          </div>
        </div>
      </CardHeader>

      {isExpanded && (
        <CardContent className="space-y-4">
          {agent.credential_fields.map((field) => (
            <div key={field.name} className="space-y-2">
              <Label htmlFor={field.name}>
                {field.label}
                {field.required && <span className="text-red-500 ml-1">*</span>}
              </Label>
              <p className="text-sm text-muted-foreground">{field.description}</p>
              <div className="flex gap-2">
                <Input
                  id={field.name}
                  type={showValues[field.name] ? "text" : field.type}
                  placeholder={field.placeholder || `Enter ${field.label.toLowerCase()}`}
                  value={credentials[field.name] || ""}
                  onChange={(e) => onCredentialChange(field.name, e.target.value)}
                  className="flex-1"
                />
                {field.type === "password" && (
                  <Button
                    type="button"
                    variant="outline"
                    size="icon"
                    onClick={() => onToggleShowValue(field.name)}
                  >
                    {showValues[field.name] ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                  </Button>
                )}
              </div>
            </div>
          ))}

          <div className="flex gap-2 pt-4">
            <Button onClick={onSave} disabled={isSaving} className="flex-1">
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-2" />
                  Save Credentials
                </>
              )}
            </Button>
            {agent.is_configured && (
              <Button onClick={onDelete} variant="destructive">
                <Trash2 className="h-4 w-4 mr-2" />
                Delete
              </Button>
            )}
          </div>

          {agent.is_configured && (
            <div className="text-sm text-muted-foreground pt-2">
              Last updated: {agent.updated_at ? new Date(agent.updated_at).toLocaleString() : "Never"}
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}
