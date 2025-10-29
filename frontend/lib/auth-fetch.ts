// Lightweight client-side auth fetch helper for Clerk
// Uses window.Clerk to obtain a JWT for the configured template

const TEMPLATE = process.env.NEXT_PUBLIC_CLERK_JWT_TEMPLATE || 'your-backend-template';

async function getClerkToken(): Promise<string | undefined> {
	if (typeof window === 'undefined') return undefined;
	const anyWin: any = window as any;
	try {
		const session = anyWin?.Clerk?.session;
		if (!session) return undefined;
		return await session.getToken({ template: TEMPLATE });
	} catch {
		return undefined;
	}
}

export async function authFetch(url: string, options: RequestInit = {}): Promise<Response> {
	const token = await getClerkToken();
	const headers: HeadersInit = {
		...(options.headers || {}),
	};
	if (token) {
		headers['Authorization'] = `Bearer ${token}`;
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


