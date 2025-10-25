"use client"

import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { SignInButton, SignOutButton, useUser, UserButton } from "@clerk/nextjs"

export default function Navbar() {
  const { isSignedIn, user } = useUser()

  return (
    <header className="w-screen left-0 right-0 flex-shrink-0 sticky top-0 z-50 bg-white border-b border-gray-200">
      <div className="flex items-center justify-between px-6 py-3 w-full">
        {/* Left section with logo */}
        <div className="flex items-center gap-4">
          <Link href="/" className="flex items-center space-x-3">
            <Image
              src="/logo.png"
              alt="Orbimesh Logo"
              width={28}
              height={28}
              className="rounded"
              priority
            />
            <span className="text-lg font-semibold text-gray-900">Orbimesh</span>
          </Link>
        </div>

        {/* Right section with navigation and auth */}
        <div className="flex items-center space-x-8">
          {/* Navigation - hidden on small screens */}
          <nav className="hidden md:flex items-center space-x-8">
            <Link 
              href="/" 
              className="text-sm font-medium text-gray-900 hover:text-blue-600 transition-colors"
            >
              HOME
            </Link>
            <Link 
              href="/api/agents" 
              className="text-sm font-medium text-gray-900 hover:text-blue-600 transition-colors"
            >
              AGENT DIRECTORY
            </Link>
            <Link 
              href="/workflow-builder" 
              className="text-sm font-medium text-gray-900 hover:text-blue-600 transition-colors"
            >
              WORKFLOW BUILDER
            </Link>
          </nav>

          {/* Auth section */}
          <div className="flex items-center space-x-4">
            {isSignedIn ? (
              <div className="flex items-center space-x-3">
                <UserButton afterSignOutUrl="/" />
              </div>
            ) : (
              <SignInButton>
                <Button className="bg-purple-600 hover:bg-purple-700 text-white text-sm font-medium px-4 py-2 rounded-md">
                  Sign In
                </Button>
              </SignInButton>
            )}
          </div>
        </div>
      </div>
    </header>
  )
}
