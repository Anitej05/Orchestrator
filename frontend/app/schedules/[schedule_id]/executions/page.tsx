"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
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
  ArrowLeft,
  CheckCircle,
  XCircle,
  Clock,
  Loader2,
  AlertCircle,
  ChevronDown,
  ChevronRight,
} from "lucide-react";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";

interface Execution {
  execution_id: string;
  status: string;
  inputs: any;
  outputs: any;
  error: string | null;
  started_at: string | null;
  completed_at: string | null;
  duration_ms: number | null;
}

export default function ExecutionHistoryPage() {
  const router = useRouter();
  const params = useParams();
  const { getToken } = useAuth();
  const { toast } = useToast();
  const [executions, setExecutions] = useState<Execution[]>([]);
  const [loading, setLoading] = useState(true);
  const [expandedExecution, setExpandedExecution] = useState<string | null>(null);

  const scheduleId = params.schedule_id as string;

  useEffect(() => {
    loadExecutions();
  }, [scheduleId]);

  const loadExecutions = async () => {
    try {
      const token = await getToken();
      const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
      const response = await fetch(
        `${API_URL}/api/schedules/${scheduleId}/executions`,
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      if (!response.ok) throw new Error("Failed to load executions");

      const data = await response.json();
      setExecutions(data.executions || []);
    } catch (error) {
      console.error("Error loading executions:", error);
      toast({
        title: "Error",
        description: "Failed to load execution history",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case "completed":
        return <CheckCircle className="h-4 w-4 text-green-600" />;
      case "failed":
        return <XCircle className="h-4 w-4 text-red-600" />;
      case "running":
        return <Loader2 className="h-4 w-4 animate-spin text-blue-600" />;
      default:
        return <Clock className="h-4 w-4 text-gray-600" />;
    }
  };

  const getStatusBadge = (status: string) => {
    const variants: Record<string, "default" | "destructive" | "secondary" | "outline"> = {
      completed: "default",
      failed: "destructive",
      running: "secondary",
      queued: "outline",
    };

    return <Badge variant={variants[status] || "outline"}>{status}</Badge>;
  };

  const formatDateTime = (dateString: string | null) => {
    if (!dateString) return "N/A";
    const date = new Date(dateString);
    return date.toLocaleString();
  };

  const formatDuration = (ms: number | null) => {
    if (!ms) return "N/A";
    if (ms < 1000) return `${ms}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${(ms / 60000).toFixed(1)}m`;
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
      <div className="flex items-center gap-4 mb-6">
        <Button variant="ghost" size="icon" onClick={() => router.push("/schedules")}>
          <ArrowLeft className="h-4 w-4" />
        </Button>
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Execution History</h1>
          <p className="text-muted-foreground mt-1">
            View all executions for this scheduled workflow
          </p>
        </div>
      </div>

      {executions.length === 0 ? (
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <AlertCircle className="h-12 w-12 text-muted-foreground mb-4" />
            <h3 className="text-lg font-semibold mb-2">No executions yet</h3>
            <p className="text-muted-foreground text-center">
              This workflow hasn't been executed by the scheduler yet. Wait for the
              scheduled time or trigger it manually.
            </p>
          </CardContent>
        </Card>
      ) : (
        <Card>
          <CardHeader>
            <CardTitle>Execution History</CardTitle>
            <CardDescription>
              {executions.filter((e) => e.status === "completed").length} completed,{" "}
              {executions.filter((e) => e.status === "failed").length} failed,{" "}
              {executions.filter((e) => e.status === "running").length} running
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {executions.map((execution) => (
                <Collapsible
                  key={execution.execution_id}
                  open={expandedExecution === execution.execution_id}
                  onOpenChange={() =>
                    setExpandedExecution(
                      expandedExecution === execution.execution_id
                        ? null
                        : execution.execution_id
                    )
                  }
                >
                  <Card>
                    <CollapsibleTrigger asChild>
                      <CardHeader className="cursor-pointer hover:bg-muted/50 transition-colors">
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-4">
                            {expandedExecution === execution.execution_id ? (
                              <ChevronDown className="h-4 w-4" />
                            ) : (
                              <ChevronRight className="h-4 w-4" />
                            )}
                            {getStatusIcon(execution.status)}
                            <div>
                              <div className="flex items-center gap-2">
                                <span className="font-mono text-sm">
                                  {execution.execution_id.slice(0, 8)}
                                </span>
                                {getStatusBadge(execution.status)}
                              </div>
                              <p className="text-sm text-muted-foreground mt-1">
                                Started: {formatDateTime(execution.started_at)}
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="text-sm font-medium">
                              {formatDuration(execution.duration_ms)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {execution.completed_at
                                ? formatDateTime(execution.completed_at)
                                : "In progress"}
                            </p>
                          </div>
                        </div>
                      </CardHeader>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                      <CardContent className="pt-0 space-y-4">
                        {execution.inputs && Object.keys(execution.inputs).length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold mb-2">Inputs</h4>
                            <pre className="bg-muted p-3 rounded-md text-xs overflow-auto max-h-40">
                              {JSON.stringify(execution.inputs, null, 2)}
                            </pre>
                          </div>
                        )}

                        {execution.outputs && Object.keys(execution.outputs).length > 0 && (
                          <div>
                            <h4 className="text-sm font-semibold mb-2">Outputs</h4>
                            <pre className="bg-muted p-3 rounded-md text-xs overflow-auto max-h-40">
                              {JSON.stringify(execution.outputs, null, 2)}
                            </pre>
                          </div>
                        )}

                        {execution.error && (
                          <div>
                            <h4 className="text-sm font-semibold mb-2 text-destructive">
                              Error
                            </h4>
                            <pre className="bg-destructive/10 p-3 rounded-md text-xs overflow-auto max-h-40 text-destructive">
                              {execution.error}
                            </pre>
                          </div>
                        )}

                        {!execution.inputs &&
                          !execution.outputs &&
                          !execution.error &&
                          execution.status === "running" && (
                            <p className="text-sm text-muted-foreground italic">
                              Execution in progress...
                            </p>
                          )}
                      </CardContent>
                    </CollapsibleContent>
                  </Card>
                </Collapsible>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
