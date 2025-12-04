"use client"

import Link from "next/link"
import Image from "next/image"
import { Button } from "@/components/ui/button"
import { SignInButton, SignOutButton, useUser, UserButton } from "@clerk/nextjs"
import { ThemeToggle } from "./theme-toggle"
import { Bot } from "lucide-react"

export default function Navbar() {
  const { isSignedIn, user } = useUser()

  return (
    <nav className="fixed top-0 left-0 w-full z-[100] h-16 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800/30 flex items-center justify-between px-6">
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
      <div className="flex items-center gap-16">
        <Link href="/" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">HOME</Link>
        <Link href="/agents" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">AGENT DIRECTORY</Link>
        <Link href="/saved-workflows" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">WORKFLOWS</Link>
        <Link href="/schedules" className="text-sm font-medium text-gray-700 dark:text-gray-200 hover:text-blue-600 dark:hover:text-blue-400 transition-colors">SCHEDULES</Link>
      </div>
      <div className="flex items-center gap-5">
        <div className="scale-125">
          <ThemeToggle />
        </div>
        <div className="scale-110">
          <UserButton afterSignOutUrl="/sign-in" />
        </div>
      </div>
    </nav>
  );
}
