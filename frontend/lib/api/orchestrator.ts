import { getBaseUrl } from "@/lib/utils";

/**
 * Approve a pending action execution.
 */
export async function approveAction(threadId: string): Promise<void> {
    const response = await fetch(`${getBaseUrl()}/api/orchestrator/action/approve`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ thread_id: threadId }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(error.detail || "Failed to approve action");
    }
}

/**
 * Reject a pending action execution.
 */
export async function rejectAction(threadId: string, reason: string): Promise<void> {
    const response = await fetch(`${getBaseUrl()}/api/orchestrator/action/reject`, {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({
            thread_id: threadId,
            reason: reason
        }),
    });

    if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: "Unknown error" }));
        throw new Error(error.detail || "Failed to reject action");
    }
}
