"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Badge } from "@/components/ui/badge";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Play, Calendar, Webhook, Trash2, Clock, CheckCircle2, XCircle } from "lucide-react";
import { authFetch } from "@/lib/auth-fetch";
import { Textarea } from "@/components/ui/textarea";
import Navbar from "@/components/navbar"

interface Workflow {
  workflow_id: string;
  name: string;
  description: string;
  created_at: string;
  task_count?: number;
}

interface WorkflowExecution {
  execution_id: string;
  status: string;
  started_at: string;
  completed_at?: string;
}

export default function WorkflowsPage() {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedWorkflow, setSelectedWorkflow] = useState<Workflow | null>(null);
  const [executeDialogOpen, setExecuteDialogOpen] = useState(false);
  const [scheduleDialogOpen, setScheduleDialogOpen] = useState(false);
  const [webhookDialogOpen, setWebhookDialogOpen] = useState(false);
  const [inputs, setInputs] = useState<Record<string, string>>({});
  const [executing, setExecuting] = useState(false);
  const [webhookUrl, setWebhookUrl] = useState("");
  const [webhookToken, setWebhookToken] = useState("");
  const [cronExpression, setCronExpression] = useState("");
  const [scheduleInputs, setScheduleInputs] = useState("");

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      const response = await authFetch("http://localhost:8000/api/workflows");
      const data = await response.json();
      setWorkflows(data);
    } catch (error) {
      console.error("Failed to load workflows:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleExecute = async () => {
    if (!selectedWorkflow) return;

    setExecuting(true);
    try {
      // Open WebSocket connection for streaming
      const ws = new WebSocket(`ws://localhost:8000/ws/workflow/${selectedWorkflow.workflow_id}/execute`);
      
      ws.onopen = () => {
        // Send inputs with owner info
        ws.send(JSON.stringify({
          inputs,
          owner: { user_id: "current_user" } // Will be replaced by actual auth
        }));
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        console.log("Workflow event:", data);
        
        if (data.node === "__complete__") {
          setExecuteDialogOpen(false);
          setInputs({});
          ws.close();
          // Could redirect to conversation view here
        } else if (data.node === "__error__") {
          console.error("Workflow error:", data.error);
          ws.close();
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
      };
    } catch (error) {
      console.error("Failed to execute workflow:", error);
    } finally {
      setExecuting(false);
    }
  };

  const handleCreateWebhook = async () => {
    if (!selectedWorkflow) return;

    try {
      const response = await authFetch(
        `http://localhost:8000/api/workflows/${selectedWorkflow.workflow_id}/webhook`,
        { method: "POST" }
      );
      const data = await response.json();
      setWebhookUrl(`http://localhost:8000${data.webhook_url}?webhook_token=${data.webhook_token}`);
      setWebhookToken(data.webhook_token);
    } catch (error) {
      console.error("Failed to create webhook:", error);
    }
  };

  const handleCreateSchedule = async () => {
    if (!selectedWorkflow || !cronExpression.trim()) {
      alert("Please enter a cron expression");
      return;
    }

    try {
      const inputTemplate = scheduleInputs.trim() ? JSON.parse(scheduleInputs) : {};
      const response = await authFetch(
        `http://localhost:8000/api/workflows/${selectedWorkflow.workflow_id}/schedule?cron_expression=${encodeURIComponent(cronExpression)}&input_template=${encodeURIComponent(JSON.stringify(inputTemplate))}`,
        { 
          method: "POST",
          headers: { "Content-Type": "application/json" }
        }
      );

      if (response.ok) {
        alert("Schedule created successfully!");
        setScheduleDialogOpen(false);
        setCronExpression("");
        setScheduleInputs("");
      } else {
        const error = await response.json();
        alert(`Failed to create schedule: ${error.detail || "Unknown error"}`);
      }
    } catch (error: any) {
      console.error("Failed to create schedule:", error);
      alert(`Error: ${error.message || "Invalid JSON in input parameters"}`);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <p className="text-gray-500">Loading workflows...</p>
      </div>
    );
  }

  return (
    <>
      <Navbar />
      <div className="container mx-auto p-6 max-w-6xl">
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2">Saved Workflows</h1>
          <p className="text-gray-600">
            Manage and execute your saved workflows. Re-run them with different inputs, schedule executions, or trigger via webhooks.
        </p>
      </div>

      {workflows.length === 0 ? (
        <Card>
          <CardContent className="text-center py-12">
            <p className="text-gray-500 mb-4">No workflows saved yet.</p>
            <p className="text-sm text-gray-400">
              Complete a conversation and click "Save as Workflow" to create your first reusable workflow.
            </p>
          </CardContent>
        </Card>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {workflows.map((workflow) => (
            <Card key={workflow.workflow_id} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <CardTitle className="flex items-center justify-between">
                  <span className="truncate">{workflow.name}</span>
                  <Badge variant="outline">{workflow.task_count || 0} tasks</Badge>
                </CardTitle>
                <CardDescription className="line-clamp-2">
                  {workflow.description || "No description"}
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center text-sm text-gray-500">
                  <Clock className="w-4 h-4 mr-2" />
                  {new Date(workflow.created_at).toLocaleDateString()}
                </div>
                <div className="flex gap-2 flex-wrap">
                  <Button
                    size="sm"
                    onClick={() => {
                      setSelectedWorkflow(workflow);
                      setExecuteDialogOpen(true);
                    }}
                    className="flex-1"
                  >
                    <Play className="w-4 h-4 mr-2" />
                    Execute
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSelectedWorkflow(workflow);
                      setScheduleDialogOpen(true);
                    }}
                  >
                    <Calendar className="w-4 h-4" />
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => {
                      setSelectedWorkflow(workflow);
                      setWebhookDialogOpen(true);
                      handleCreateWebhook();
                    }}
                  >
                    <Webhook className="w-4 h-4" />
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Execute Dialog */}
      <Dialog open={executeDialogOpen} onOpenChange={setExecuteDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Execute Workflow</DialogTitle>
            <DialogDescription>
              {selectedWorkflow?.name}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <p className="text-sm text-gray-600">
              Provide input values for this workflow execution:
            </p>
            <div className="space-y-3">
              <div>
                <label className="text-sm font-medium">Company Name</label>
                <Input
                  placeholder="e.g., Acme Corp"
                  value={inputs.company_name || ""}
                  onChange={(e) => setInputs({ ...inputs, company_name: e.target.value })}
                />
              </div>
              <div>
                <label className="text-sm font-medium">Additional Context (optional)</label>
                <Textarea
                  placeholder="Any additional information..."
                  value={inputs.context || ""}
                  onChange={(e) => setInputs({ ...inputs, context: e.target.value })}
                  rows={3}
                />
              </div>
            </div>
          </div>
          <DialogFooter>
            <Button onClick={handleExecute} disabled={executing}>
              {executing ? "Executing..." : "Execute Workflow"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Webhook Dialog */}
      <Dialog open={webhookDialogOpen} onOpenChange={setWebhookDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Webhook Trigger</DialogTitle>
            <DialogDescription>
              Trigger this workflow from external systems
            </DialogDescription>
          </DialogHeader>
          {webhookUrl && (
            <div className="space-y-4 py-4">
              <div>
                <label className="text-sm font-medium">Webhook URL</label>
                <Input value={webhookUrl} readOnly className="font-mono text-xs" />
              </div>
              <div>
                <label className="text-sm font-medium">Token</label>
                <Input value={webhookToken} readOnly className="font-mono text-xs" />
              </div>
              <div className="p-3 bg-gray-50 rounded-md">
                <p className="text-xs text-gray-600 mb-2">Example cURL request:</p>
                <code className="text-xs block p-2 bg-white rounded border">
                  curl -X POST "{webhookUrl}" \<br />
                  &nbsp;&nbsp;-H "Content-Type: application/json" \<br />
                  &nbsp;&nbsp;-d '{`{"company_name": "Example Corp"}`}'
                </code>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>

      {/* Schedule Dialog */}
      <Dialog open={scheduleDialogOpen} onOpenChange={setScheduleDialogOpen}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Schedule Workflow</DialogTitle>
            <DialogDescription>
              Set up automated execution with cron expression
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 py-4">
            <div>
              <label className="text-sm font-medium">Cron Expression</label>
              <Input 
                value={cronExpression}
                onChange={(e) => setCronExpression(e.target.value)}
                placeholder="0 9 * * 1-5" 
                className="font-mono"
                title="Format: minute hour day month day_of_week (e.g., 0 9 * * 1-5 for weekdays at 9 AM)"
              />
              <p className="text-xs text-gray-500 mt-1">
                Examples: <code>0 9 * * *</code> (daily 9 AM), <code>*/30 * * * *</code> (every 30 min)
              </p>
            </div>
            <div>
              <label className="text-sm font-medium">Input Parameters (JSON)</label>
              <Textarea 
                value={scheduleInputs}
                onChange={(e) => setScheduleInputs(e.target.value)}
                placeholder='{"company_name": "Tesla", "context": "Q4 earnings"}'
                className="font-mono text-xs"
                rows={4}
              />
            </div>
          </div>
          <DialogFooter>
            <Button onClick={handleCreateSchedule} disabled={!cronExpression.trim()}>
              Create Schedule
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
    </>
  );
}
