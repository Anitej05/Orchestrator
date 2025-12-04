"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@clerk/nextjs";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Clock, Calendar, Loader2, Info } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";

interface ScheduleWorkflowDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  workflowId: string;
  workflowName: string;
  onScheduleCreated?: () => void;
}

type ScheduleType = "hourly" | "daily" | "weekly" | "monthly";

export function ScheduleWorkflowDialog({
  open,
  onOpenChange,
  workflowId,
  workflowName,
  onScheduleCreated,
}: ScheduleWorkflowDialogProps) {
  const { getToken } = useAuth();
  const { toast } = useToast();
  const [loading, setLoading] = useState(false);
  
  // Schedule configuration
  const [scheduleType, setScheduleType] = useState<ScheduleType>("daily");
  const [hour, setHour] = useState("09");
  const [minute, setMinute] = useState("00");
  const [dayOfWeek, setDayOfWeek] = useState("1"); // Monday
  const [dayOfMonth, setDayOfMonth] = useState("1");
  
  const [inputTemplate, setInputTemplate] = useState("{}");
  const [nextRuns, setNextRuns] = useState<string[]>([]);

  // Generate cron expression from user-friendly inputs
  // Convert IST (UTC+5:30) to UTC for cron scheduling
  const generateCronExpression = (): string => {
    const userHour = parseInt(hour) || 9;
    const userMinute = parseInt(minute) || 0;
    
    // Convert IST time to UTC
    // IST is UTC+5:30, so subtract 5:30 from user's IST time to get UTC time
    let utcHour = userHour - 5;
    let utcMinute = userMinute - 30;
    
    // Handle minute underflow
    if (utcMinute < 0) {
      utcHour -= 1;
      utcMinute += 60;
    }
    
    // Handle hour underflow (previous day)
    if (utcHour < 0) {
      utcHour += 24;
    }
    
    const m = utcMinute.toString().padStart(2, '0');
    const h = utcHour.toString().padStart(2, '0');
    
    switch (scheduleType) {
      case "hourly":
        return `${m} * * * *`; // Every hour at specified minute (UTC)
      case "daily":
        return `${m} ${h} * * *`; // Every day at specified UTC time (user's IST time)
      case "weekly":
        return `${m} ${h} * * ${dayOfWeek}`; // Specific day of week at specified UTC time
      case "monthly":
        return `${m} ${h} ${dayOfMonth} * *`; // Specific day of month at specified UTC time
      default:
        return `0 9 * * *`;
    }
  };

  const cronExpression = generateCronExpression();

  useEffect(() => {
    calculateNextRuns();
  }, [scheduleType, hour, minute, dayOfWeek, dayOfMonth]);

  const calculateNextRuns = () => {
    try {
      const now = new Date();
      const runs: string[] = [];
      const h = parseInt(hour) || 9;
      const m = parseInt(minute) || 0;

      for (let i = 0; i < 5; i++) {
        const nextRun = new Date(now);
        
        switch (scheduleType) {
          case "hourly":
            nextRun.setHours(now.getHours() + i + 1);
            nextRun.setMinutes(m);
            break;
          case "daily":
            nextRun.setDate(now.getDate() + i);
            nextRun.setHours(h);
            nextRun.setMinutes(m);
            break;
          case "weekly":
            const targetDay = parseInt(dayOfWeek);
            const currentDay = nextRun.getDay();
            let daysUntil = (targetDay - currentDay + 7) % 7;
            if (daysUntil === 0 && i > 0) daysUntil = 7;
            nextRun.setDate(nextRun.getDate() + daysUntil + (i * 7));
            nextRun.setHours(h);
            nextRun.setMinutes(m);
            break;
          case "monthly":
            const targetDate = parseInt(dayOfMonth);
            nextRun.setMonth(nextRun.getMonth() + i);
            nextRun.setDate(targetDate);
            nextRun.setHours(h);
            nextRun.setMinutes(m);
            break;
        }
        
        nextRun.setSeconds(0);
        nextRun.setMilliseconds(0);
        
        if (nextRun > now) {
          runs.push(nextRun.toLocaleString());
        }
      }

      setNextRuns(runs.slice(0, 5));
    } catch (error) {
      console.error("Error calculating next runs:", error);
      setNextRuns([]);
    }
  };

  const validateInputTemplate = () => {
    if (!inputTemplate.trim() || inputTemplate.trim() === "{}") return true;
    try {
      JSON.parse(inputTemplate);
      return true;
    } catch {
      return false;
    }
  };

  const handleSchedule = async () => {
    if (!validateInputTemplate()) {
      toast({
        title: "Invalid JSON",
        description: "Input template must be valid JSON",
        variant: "destructive",
      });
      return;
    }

    setLoading(true);
    try {
      const token = await getToken();
      const template = inputTemplate.trim() && inputTemplate.trim() !== "{}" ? JSON.parse(inputTemplate) : {};
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

      const response = await fetch(
        `${API_URL}/api/workflows/${workflowId}/schedule`,
        {
          method: "POST",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            cron_expression: cronExpression,
            input_template: template,
          }),
        }
      );

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Failed to schedule workflow" }));
        throw new Error(error.detail || "Failed to schedule workflow");
      }

      toast({
        title: "Workflow scheduled",
        description: `${workflowName} will run automatically according to schedule`,
      });

      onScheduleCreated?.();
      onOpenChange(false);

      // Reset form
      setScheduleType("daily");
      setHour("09");
      setMinute("00");
      setInputTemplate("{}");
    } catch (error: any) {
      console.error("Error scheduling workflow:", error);
      toast({
        title: "Error",
        description: error.message || "Failed to schedule workflow",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Schedule Workflow</DialogTitle>
          <DialogDescription>
            Set up automatic execution for <strong>{workflowName}</strong>
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 py-4">
          {/* Schedule Frequency */}
          <div className="space-y-3">
            <Label>How often should this workflow run?</Label>
            <RadioGroup value={scheduleType} onValueChange={(value) => setScheduleType(value as ScheduleType)}>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="hourly" id="hourly" />
                <Label htmlFor="hourly" className="font-normal cursor-pointer">
                  Hourly - Runs every hour
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="daily" id="daily" />
                <Label htmlFor="daily" className="font-normal cursor-pointer">
                  Daily - Runs once per day
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="weekly" id="weekly" />
                <Label htmlFor="weekly" className="font-normal cursor-pointer">
                  Weekly - Runs once per week
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="monthly" id="monthly" />
                <Label htmlFor="monthly" className="font-normal cursor-pointer">
                  Monthly - Runs once per month
                </Label>
              </div>
            </RadioGroup>
          </div>

          {/* Time Selection */}
          {scheduleType !== "hourly" && (
            <div className="space-y-2">
              <Label>What time should it run? (IST - Asia/Kolkata)</Label>
              <div className="flex items-center gap-3">
                <div className="flex-1">
                  <Label htmlFor="hour" className="text-xs text-muted-foreground">Hour (0-23 IST)</Label>
                  <Input
                    id="hour"
                    type="number"
                    min="0"
                    max="23"
                    value={hour}
                    onChange={(e) => setHour(e.target.value)}
                    className="text-center"
                  />
                </div>
                <span className="text-2xl mt-5">:</span>
                <div className="flex-1">
                  <Label htmlFor="minute" className="text-xs text-muted-foreground">Minute (0-59)</Label>
                  <Input
                    id="minute"
                    type="number"
                    min="0"
                    max="59"
                    value={minute}
                    onChange={(e) => setMinute(e.target.value)}
                    className="text-center"
                  />
                </div>
                <div className="flex-1 flex items-end">
                  <div className="text-muted-foreground text-sm px-3 py-2 border rounded-md bg-muted">
                    {hour.padStart(2, '0')}:{minute.padStart(2, '0')} IST
                  </div>
                </div>
              </div>
              <p className="text-xs text-muted-foreground">
                ✓ All times are in IST (UTC+5:30). Times are automatically converted to UTC for backend scheduling.
              </p>
            </div>
          )}

          {/* Minute for hourly */}
          {scheduleType === "hourly" && (
            <div className="space-y-2">
              <Label htmlFor="minute-hourly">At which minute of the hour?</Label>
              <div className="flex items-center gap-3">
                <Input
                  id="minute-hourly"
                  type="number"
                  min="0"
                  max="59"
                  value={minute}
                  onChange={(e) => setMinute(e.target.value)}
                  className="w-24 text-center"
                />
                <span className="text-sm text-muted-foreground">
                  (e.g., 30 means it runs at xx:30 every hour)
                </span>
              </div>
            </div>
          )}

          {/* Day of Week Selection */}
          {scheduleType === "weekly" && (
            <div className="space-y-2">
              <Label htmlFor="day-of-week">Which day of the week?</Label>
              <Select value={dayOfWeek} onValueChange={setDayOfWeek}>
                <SelectTrigger id="day-of-week">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="0">Sunday</SelectItem>
                  <SelectItem value="1">Monday</SelectItem>
                  <SelectItem value="2">Tuesday</SelectItem>
                  <SelectItem value="3">Wednesday</SelectItem>
                  <SelectItem value="4">Thursday</SelectItem>
                  <SelectItem value="5">Friday</SelectItem>
                  <SelectItem value="6">Saturday</SelectItem>
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Day of Month Selection */}
          {scheduleType === "monthly" && (
            <div className="space-y-2">
              <Label htmlFor="day-of-month">Which day of the month?</Label>
              <Select value={dayOfMonth} onValueChange={setDayOfMonth}>
                <SelectTrigger id="day-of-month">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent className="max-h-60">
                  {Array.from({ length: 31 }, (_, i) => i + 1).map((day) => (
                    <SelectItem key={day} value={day.toString()}>
                      {day}{day === 1 ? 'st' : day === 2 ? 'nd' : day === 3 ? 'rd' : 'th'}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          )}

          {/* Preview Next Runs */}
          {nextRuns.length > 0 && (
            <div className="space-y-2">
              <Label className="flex items-center gap-2">
                <Calendar className="h-4 w-4" />
                Next 5 Scheduled Runs
              </Label>
              <div className="bg-muted p-3 rounded-md space-y-1">
                {nextRuns.map((run, idx) => (
                  <div key={idx} className="flex items-center gap-2 text-sm">
                    <Clock className="h-3 w-3 text-muted-foreground" />
                    <span>{run}</span>
                    {idx === 0 && (
                      <Badge variant="secondary" className="text-xs">
                        Next
                      </Badge>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Technical Details */}
          <div className="bg-blue-50 dark:bg-blue-950 border border-blue-200 dark:border-blue-800 rounded-md p-3">
            <div className="flex items-start gap-2">
              <Info className="h-4 w-4 text-blue-600 dark:text-blue-400 mt-0.5" />
              <div className="space-y-1">
                <p className="text-sm text-blue-900 dark:text-blue-100">
                  <strong>Cron Expression:</strong> <code className="bg-blue-100 dark:bg-blue-900 px-2 py-0.5 rounded">{cronExpression}</code>
                </p>
                <p className="text-xs text-blue-700 dark:text-blue-300">
                  All times are in UTC timezone
                </p>
              </div>
            </div>
          </div>

          {/* Input Template */}
          <div className="space-y-2">
            <Label htmlFor="input-template">
              Input Parameters (JSON)
              <span className="text-xs text-muted-foreground ml-2">(optional)</span>
            </Label>
            <Textarea
              id="input-template"
              placeholder='{"param1": "value1", "param2": "value2"}'
              value={inputTemplate}
              onChange={(e) => setInputTemplate(e.target.value)}
              rows={4}
              className="font-mono text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Leave as {} if your workflow doesn't need input parameters
            </p>
            {inputTemplate && inputTemplate !== "{}" && !validateInputTemplate() && (
              <p className="text-xs text-destructive">Invalid JSON format</p>
            )}
          </div>
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={loading}>
            Cancel
          </Button>
          <Button onClick={handleSchedule} disabled={loading || !validateInputTemplate()}>
            {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
            Schedule Workflow
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
