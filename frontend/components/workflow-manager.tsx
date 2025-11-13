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
      const response = await authFetch("http://localhost:8000/api/workflows");
      const data = await response.json();
      setWorkflows(data);
    } catch (error) {
      console.error("Failed to load workflows:", error);
    }
  };

  const saveAsWorkflow = async () => {
    if (!threadId || !saveName) return;
    
    try {
      await authFetch(`http://localhost:8000/api/workflows?thread_id=${threadId}&name=${encodeURIComponent(saveName)}&description=${encodeURIComponent(saveDesc)}`, {
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
        <Card>
          <CardHeader>
            <CardTitle>Save as Workflow</CardTitle>
          </CardHeader>
          <CardContent className="space-y-2">
            <Input
              placeholder="Workflow name"
              value={saveName}
              onChange={(e) => setSaveName(e.target.value)}
            />
            <Textarea
              placeholder="Description"
              value={saveDesc}
              onChange={(e) => setSaveDesc(e.target.value)}
            />
            <Button onClick={saveAsWorkflow}>Save</Button>
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>My Workflows</CardTitle>
        </CardHeader>
        <CardContent>
          {workflows.map((w) => (
            <div key={w.workflow_id} className="p-2 border rounded mb-2">
              <h3 className="font-bold">{w.name}</h3>
              <p className="text-sm text-gray-600">{w.description}</p>
              <Button 
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
