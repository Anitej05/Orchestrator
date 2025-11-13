"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Save, Check } from "lucide-react";
import { authFetch } from "@/lib/auth-fetch";

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
        setSaved(true);
        setTimeout(() => {
          setOpen(false);
          setSaved(false);
          setName("");
          setDescription("");
        }, 1500);
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

  return (
    <Dialog open={open} onOpenChange={setOpen}>
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
            {saved ? (
              <>
                <Check className="w-4 h-4" />
                Saved!
              </>
            ) : saving ? (
              "Saving..."
            ) : (
              <>
                <Save className="w-4 h-4" />
                Save Workflow
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
