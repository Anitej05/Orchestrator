"use client";

import { useState, useEffect } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { authFetch } from "@/lib/auth-fetch";

interface Workflow {
  workflow_id: string;
  name: string;
  description: string;
  created_at: string;
}

export default function WorkflowManager({ threadId }: { threadId?: string }) {
  const [workflows, setWorkflows] = useState<Workflow[]>([]);
  const [saveName, setSaveName] = useState("");
  const [saveDesc, setSaveDesc] = useState("");
  const [selectedWorkflow, setSelectedWorkflow] = useState<string | null>(null);
  const [inputs, setInputs] = useState<Record<string, string>>({});

  useEffect(() => {
    loadWorkflows();
  }, []);

  const loadWorkflows = async () => {
    try {
      import { API_BASE_URL } from '@/lib/config';
      const response = await authFetch(`${API_BASE_URL}/api/workflows`);
      const data = await response.json();
      setWorkflows(data);
    } catch (error) {
      console.error("Failed to load workflows:", error);
    }
  };

  const saveAsWorkflow = async () => {
    if (!threadId || !saveName) return;
    
    try {
      await authFetch(`${API_BASE_URL}/api/workflows?thread_id=${threadId}&name=${encodeURIComponent(saveName)}&description=${encodeURIComponent(saveDesc)}`, {
        method: "POST"
      });
      setSaveName("");
      setSaveDesc("");
      loadWorkflows();
    } catch (error) {
      console.error("Failed to save workflow:", error);
    }
  };

  const executeWorkflow = async (workflowId: string) => {
    // Frontend should open WebSocket to /ws/workflow/{workflowId}/execute
    // and send inputs via the connection
    console.log("Execute workflow:", workflowId, "with inputs:", inputs);
  };

  return (
    <div className="space-y-4">
      {threadId && (
        <Card className="ui-card">
          <CardHeader>
            <CardTitle className="ui-section-header">Save as Workflow</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Input
              className="ui-input"
              placeholder="Workflow name"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
            />
            <Textarea
              className="ui-textarea"
              placeholder="Description"
              value={saveDesc}
              onChange={(e) => setSaveDesc(e.target.value)}
            />
            <Button variant="ui-primary" onClick={saveAsWorkflow}>Save</Button>
          </CardContent>
        </Card>
      )}

      <Card className="ui-card">
        <CardHeader>
          <CardTitle className="ui-section-header">My Workflows</CardTitle>
        </CardHeader>
        <CardContent>
          {workflows.map((w) => (
            <div key={w.workflow_id} className="ui-card-hover p-2 mb-2">
              <h3 className="ui-task-name">{w.name}</h3>
              <p className="ui-task-description">{w.description}</p>
              <Button 
                variant="ui-secondary"
                size="sm" 
                onClick={() => setSelectedWorkflow(w.workflow_id)}
                className="mt-2"
              >
                Execute
              </Button>
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

