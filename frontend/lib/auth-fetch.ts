// Lightweight client-side auth fetch helper for Clerk
// Uses window.Clerk to obtain a JWT for the configured template

const TEMPLATE = process.env.NEXT_PUBLIC_CLERK_JWT_TEMPLATE || 'your-backend-template';

async function getClerkToken(): Promise<string | undefined> {
	if (typeof window === 'undefined') {
		console.warn('[authFetch] Running on server-side, no token available');
		return undefined;
	}
	const anyWin: any = window as any;
	try {
		const clerk = anyWin?.Clerk;
		if (!clerk) {
			console.error('[authFetch] window.Clerk is not available');
			return undefined;
		}

		const session = clerk.session;
		if (!session) {
			console.error('[authFetch] No active Clerk session found');
			return undefined;
		}

		// Try with template first if configured
		if (TEMPLATE && TEMPLATE !== 'your-backend-template') {
			console.log('[authFetch] Attempting to get token with template:', TEMPLATE);
			try {
				const token = await session.getToken({ template: TEMPLATE });
				if (token) {
					console.log('[authFetch] Token retrieved successfully with template, length:', token.length);
					return token;
				}
			} catch (templateError) {
				console.warn('[authFetch] Failed to get token with template, trying default:', templateError);
			}
		}

		// Fallback: try without template (default Clerk JWT)
		console.log('[authFetch] Getting default Clerk token without template');
		const token = await session.getToken();

		if (token) {
			console.log('[authFetch] Default token retrieved successfully, length:', token.length);
		} else {
			console.error('[authFetch] Token is null/undefined after getToken call');
		}

		return token;
	} catch (error) {
		console.error('[authFetch] Error getting Clerk token:', error);
		return undefined;
	}
}

export async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
	const token = await getClerkToken();
	const headers: Record<string, string> = {
		...(options.headers as Record<string, string> || {}),
	};

	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
		console.log('[authFetch] Request to', url, 'with Authorization header');
	} else {
		console.warn('[authFetch] No token available for request to', url);
	}

	try {
		const response = await fetch(url, { ...options, headers });
		return response;
	} catch (error) {
		console.error('[authFetch] Network error for', url, ':', error);
		// Re-throw with more context
		throw new Error(`Network request failed for ${url}: ${error instanceof Error ? error.message : 'Unknown error'}`);
	}
}




