"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Save, Check, Calendar } from "lucide-react";
import { authFetch } from "@/lib/auth-fetch";
import { ScheduleWorkflowDialog } from "@/components/schedule-workflow-dialog";

interface SaveWorkflowButtonProps {
  threadId: string | null;
  disabled?: boolean;
}

export default function SaveWorkflowButton({ threadId, disabled }: SaveWorkflowButtonProps) {
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [savedWorkflowId, setSavedWorkflowId] = useState<string | null>(null);
  const [showScheduleDialog, setShowScheduleDialog] = useState(false);

  const handleSave = async () => {
    if (!threadId || !name.trim()) return;

    setSaving(true);
    try {
      const response = await authFetch(
        `http://localhost:8000/api/workflows?thread_id=${threadId}&name=${encodeURIComponent(name)}&description=${encodeURIComponent(description)}`,
        { 
          method: "POST",
          headers: {
            "Content-Type": "application/json"
          }
        }
      );

      if (response.ok) {
        const data = await response.json();
        setSaved(true);
        setSavedWorkflowId(data.workflow_id);
        // Keep dialog open to show schedule option
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error("Failed to save workflow:", response.status, errorData);
        alert(`Failed to save workflow: ${errorData.detail || response.statusText}`);
      }
    } catch (error) {
      console.error("Failed to save workflow:", error);
      alert(`Network error: Unable to connect to backend. Please ensure the backend server is running.`);
    } finally {
      setSaving(false);
    }
  };

  const handleScheduleNow = () => {
    setOpen(false);
    setShowScheduleDialog(true);
  };

  const handleClose = () => {
    setOpen(false);
    setSaved(false);
    setName("");
    setDescription("");
    setSavedWorkflowId(null);
  };

  return (
    <>
      <Dialog open={open} onOpenChange={(isOpen) => { 
        if (!isOpen) handleClose();
        else setOpen(isOpen);
      }}>
        <DialogTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            disabled={disabled || !threadId}
            className="gap-2"
          >
            <Save className="w-4 h-4" />
            Save as Workflow
          </Button>
        </DialogTrigger>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Save as Workflow</DialogTitle>
            <DialogDescription>
              Save this conversation as a reusable workflow. You can re-run it with different inputs later.
            </DialogDescription>
          </DialogHeader>
          
          {!saved ? (
            <>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Workflow Name *</label>
                  <Input
                    placeholder="e.g., Company Research & Outreach"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    disabled={saving}
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Description</label>
                  <Textarea
                    placeholder="Describe what this workflow does..."
                    value={description}
                    onChange={(e) => setDescription(e.target.value)}
                    disabled={saving}
                    rows={3}
                  />
                </div>
              </div>
              <DialogFooter>
                <Button
                  onClick={handleSave}
                  disabled={!name.trim() || saving}
                  className="gap-2"
                >
                  {saving ? (
                    "Saving..."
                  ) : (
                    <>
                      <Save className="w-4 h-4" />
                      Save Workflow
                    </>
                  )}
                </Button>
              </DialogFooter>
            </>
          ) : (
            <>
              <div className="py-6 text-center">
                <div className="flex justify-center mb-4">
                  <div className="w-12 h-12 bg-green-100 rounded-full flex items-center justify-center">
                    <Check className="w-6 h-6 text-green-600" />
                  </div>
                </div>
                <h3 className="text-lg font-semibold mb-2">Workflow Saved!</h3>
                <p className="text-sm text-muted-foreground mb-6">
                  Your workflow has been saved successfully. Would you like to schedule it to run automatically?
                </p>
              </div>
              <DialogFooter className="flex gap-2">
                <Button variant="outline" onClick={handleClose}>
                  Close
                </Button>
                <Button onClick={handleScheduleNow} className="gap-2">
                  <Calendar className="w-4 h-4" />
                  Schedule Now
                </Button>
              </DialogFooter>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Schedule Dialog */}
      {savedWorkflowId && (
        <ScheduleWorkflowDialog
          open={showScheduleDialog}
          onOpenChange={setShowScheduleDialog}
          workflowId={savedWorkflowId}
          workflowName={name}
          onScheduleCreated={() => {
            setShowScheduleDialog(false);
            // Reset all state
            setSaved(false);
            setName("");
            setDescription("");
            setSavedWorkflowId(null);
          }}
        />
      )}
    </>
  );
}
