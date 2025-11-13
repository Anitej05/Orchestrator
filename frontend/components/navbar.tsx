"use client"

import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { SignInButton, SignOutButton, useUser, UserButton } from "@clerk/nextjs"
import DarkModeToggle from "./dark-mode-toggle"
import { Bot } from "lucide-react"

export default function Navbar() {
  const { isSignedIn, user } = useUser()

  return (
    <nav className="fixed top-0 left-0 w-full z-[100] h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800 flex items-center justify-between px-6">
      <div className="flex items-center gap-3">
        <Link href="/" className="flex items-center gap-2">
          <Image
            src="/logo.png"
            alt="Orbimesh Logo"
            width={32}
            height={32}
            className="rounded"
            priority
          />
          <span className="text-lg font-bold text-blue-600 dark:text-blue-400">Orbimesh</span>
        </Link>
      </div>
      <div className="flex items-center gap-6">
        <Link href="/" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400">HOME</Link>
        <Link href="/agents" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400">AGENT DIRECTORY</Link>
        <Link href="/saved-workflows" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400">SAVED WORKFLOWS</Link>
        <Link href="/workflow" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400">WORKFLOW BUILDER</Link>
      </div>
      <div className="flex items-center gap-4">
        <DarkModeToggle />
        <UserButton afterSignOutUrl="/sign-in" />
      </div>
    </nav>
  );
}
