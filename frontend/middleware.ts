import { NextRequest, NextResponse } from 'next/server'

export function middleware(request: NextRequest) {
  // Handle Chrome DevTools requests gracefully
  if (request.nextUrl.pathname.startsWith('/.well-known/appspecific/')) {
    return new NextResponse(null, { status: 404 })
  }

  return NextResponse.next()
}

export const config = {
  matcher: '/.well-known/appspecific/:path*'
}
