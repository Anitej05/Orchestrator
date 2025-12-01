"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@clerk/nextjs";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { useToast } from "@/hooks/use-toast";
import {
  Clock,
  Pause,
  Play,
  Trash2,
  History,
  Calendar,
  Loader2,
  AlertCircle,
} from "lucide-react";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface Schedule {
  schedule_id: string;
  workflow_id: string;
  workflow_name: string;
  cron_expression: string;
  input_template: any;
  is_active: boolean;
  last_run_at: string | null;
  next_run_at: string | null;
  created_at: string;
}

export default function SchedulesPage() {
  const router = useRouter();
  const { getToken } = useAuth();
  const { toast } = useToast();
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [loading, setLoading] = useState(true);
  const [deleteScheduleId, setDeleteScheduleId] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState<string | null>(null);

  useEffect(() => {
    loadSchedules();
  }, []);

  const loadSchedules = async () => {
    try {
      const token = await getToken();
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(`${API_URL}/api/schedules`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });

      if (!response.ok) throw new Error("Failed to load schedules");

      const data = await response.json();
      setSchedules(data.schedules || []);
    } catch (error) {
      console.error("Error loading schedules:", error);
      toast({
        title: "Error",
        description: "Failed to load scheduled workflows",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const toggleSchedule = async (scheduleId: string, currentActive: boolean) => {
    setActionLoading(scheduleId);
    try {
      const token = await getToken();
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(
        `${API_URL}/api/schedules/${scheduleId}`,
        {
          method: "PATCH",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ is_active: !currentActive }),
        }
      );

      if (!response.ok) throw new Error("Failed to update schedule");

      toast({
        title: currentActive ? "Schedule paused" : "Schedule resumed",
        description: currentActive
          ? "The workflow will no longer run automatically"
          : "The workflow will run according to schedule",
      });

      await loadSchedules();
    } catch (error) {
      console.error("Error toggling schedule:", error);
      toast({
        title: "Error",
        description: "Failed to update schedule",
        variant: "destructive",
      });
    } finally {
      setActionLoading(null);
    }
  };

  const deleteSchedule = async (scheduleId: string, workflowId: string) => {
    setActionLoading(scheduleId);
    try {
      const token = await getToken();
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(
        `${API_URL}/api/workflows/${workflowId}/schedule/${scheduleId}`,
        {
          method: "DELETE",
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) throw new Error("Failed to delete schedule");

      toast({
        title: "Schedule deleted",
        description: "The workflow schedule has been removed",
      });

      await loadSchedules();
    } catch (error) {
      console.error("Error deleting schedule:", error);
      toast({
        title: "Error",
        description: "Failed to delete schedule",
        variant: "destructive",
      });
    } finally {
      setActionLoading(null);
      setDeleteScheduleId(null);
    }
  };

  const formatCron = (cron: string) => {
    // Convert cron to human-readable format
    const parts = cron.split(" ");
    if (parts.length !== 5) return cron;

    const [minute, hour, day, month, dayOfWeek] = parts;

    if (minute === "0" && hour === "*" && day === "*" && month === "*" && dayOfWeek === "*") {
      return "Every hour";
    }
    if (minute === "0" && hour !== "*" && day === "*" && month === "*" && dayOfWeek === "*") {
      return `Daily at ${hour}:00`;
    }
    if (minute === "0" && hour !== "*" && day === "*" && month === "*" && dayOfWeek !== "*") {
      const days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
      return `Every ${days[parseInt(dayOfWeek)]} at ${hour}:00`;
    }

    return cron;
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return "Never";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const getRelativeTime = (dateString: string | null) => {
    if (!dateString) return null;
    const date = new Date(dateString);
    const now = new Date();
    const diff = date.getTime() - now.getTime();

    if (diff < 0) return "Overdue";

    const minutes = Math.floor(diff / 60000);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);

    if (days > 0) return `in ${days} day${days > 1 ? "s" : ""}`;
    if (hours > 0) return `in ${hours} hour${hours > 1 ? "s" : ""}`;
    if (minutes > 0) return `in ${minutes} minute${minutes > 1 ? "s" : ""}`;
    return "soon";
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
      </div>
    );
  }

  return (
    <div className="container mx-auto py-8 px-4">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Scheduled Workflows</h1>
          <p className="text-muted-foreground mt-1">
            Manage automated workflow executions
          </p>
        </div>
        <Button onClick={() => router.push("/workflows")}>
          <Calendar className="mr-2 h-4 w-4" />
          View Workflows
        </Button>
      </div>

      {schedules.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No scheduled workflows</h3>
            <p className="text-muted-foreground text-center mb-4">
              You haven't scheduled any workflows yet. Schedule a workflow to run it
              automatically.
            </p>
            <Button onClick={() => router.push("/workflows")}>
              Browse Workflows
            </Button>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Active Schedules</CardTitle>
            <CardDescription>
              {schedules.filter((s) => s.is_active).length} active,{" "}
              {schedules.filter((s) => !s.is_active).length} paused
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Workflow</TableHead>
                  <TableHead>Schedule</TableHead>
                  <TableHead>Status</TableHead>
                  <TableHead>Last Run</TableHead>
                  <TableHead>Next Run</TableHead>
                  <TableHead className="text-right">Actions</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {schedules.map((schedule) => (
                  <TableRow key={schedule.schedule_id}>
                    <TableCell className="font-medium">
                      {schedule.workflow_name}
                    </TableCell>
                    <TableCell>
                      <div className="flex items-center gap-2">
                        <Clock className="h-4 w-4 text-muted-foreground" />
                        <span className="text-sm">{formatCron(schedule.cron_expression)}</span>
                      </div>
                    </TableCell>
                    <TableCell>
                      <Badge variant={schedule.is_active ? "default" : "secondary"}>
                        {schedule.is_active ? "Active" : "Paused"}
                      </Badge>
                    </TableCell>
                    <TableCell className="text-sm text-muted-foreground">
                      {formatDateTime(schedule.last_run_at)}
                    </TableCell>
                    <TableCell>
                      {schedule.is_active && schedule.next_run_at ? (
                        <div className="flex flex-col">
                          <span className="text-sm">{formatDateTime(schedule.next_run_at)}</span>
                          <span className="text-xs text-muted-foreground">
                            {getRelativeTime(schedule.next_run_at)}
                          </span>
                        </div>
                      ) : (
                        <span className="text-sm text-muted-foreground">-</span>
                      )}
                    </TableCell>
                    <TableCell className="text-right">
                      <div className="flex items-center justify-end gap-2">
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() =>
                            router.push(`/schedules/${schedule.schedule_id}/executions`)
                          }
                          title="View execution history"
                        >
                          <History className="h-4 w-4" />
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() =>
                            toggleSchedule(schedule.schedule_id, schedule.is_active)
                          }
                          disabled={actionLoading === schedule.schedule_id}
                        >
                          {actionLoading === schedule.schedule_id ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : schedule.is_active ? (
                            <Pause className="h-4 w-4" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                        </Button>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => setDeleteScheduleId(schedule.schedule_id)}
                          disabled={actionLoading === schedule.schedule_id}
                        >
                          <Trash2 className="h-4 w-4 text-destructive" />
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </CardContent>
        </Card>
      )}

      <AlertDialog
        open={deleteScheduleId !== null}
        onOpenChange={() => setDeleteScheduleId(null)}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete schedule?</AlertDialogTitle>
            <AlertDialogDescription>
              This will permanently delete the schedule. The workflow itself will not be
              deleted and can be scheduled again later.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => {
                const schedule = schedules.find((s) => s.schedule_id === deleteScheduleId);
                if (schedule) {
                  deleteSchedule(schedule.schedule_id, schedule.workflow_id);
                }
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
