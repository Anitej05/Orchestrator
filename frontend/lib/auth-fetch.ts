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
	
	return fetch(url, { ...options, headers });
}

export async function getOwnerFromClerk(): Promise<{ user_id: string; email?: string } | undefined> {
	if (typeof window === 'undefined') return undefined;
	const anyWin: any = window as any;
	try {
		const user = anyWin?.Clerk?.user;
		if (!user) return undefined;
		const email = user?.primaryEmailAddress?.emailAddress;
		return { user_id: user.id, email };
	} catch {
		return undefined;
	}
}


