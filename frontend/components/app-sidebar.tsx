// components/app-sidebar.tsx
"use client"

import Link from "next/link"
import Image from "next/image"
import { usePathname, useRouter } from "next/navigation"
import { useState, useEffect } from "react"
import { UserButtonWrapper } from "@/components/user-button-wrapper"
import {
  SidebarContent,
  SidebarHeader,
  SidebarGroup,
  SidebarGroupContent,
  useSidebar,
} from "@/components/ui/sidebar"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Plus, Users, Workflow, BarChart3, History, X, Link2, Loader2, CalendarClock, Save, Settings } from "lucide-react"
import ConversationsDropdown from "./conversations-dropdown"
import { ThemeToggle } from "./theme-toggle"
import { cn } from "@/lib/utils"
import { useNewConversation } from "@/hooks/use-new-conversation"

interface AppSidebarProps {
  onConversationSelect?: (threadId: string) => void;
  onNewConversation?: () => void;
  currentThreadId?: string;
}

export default function AppSidebar({ onConversationSelect, onNewConversation, currentThreadId }: AppSidebarProps) {
  const { open, setOpen, toggleSidebar } = useSidebar()
  const pathname = usePathname()
  const router = useRouter()
  const [navigatingTo, setNavigatingTo] = useState<string | null>(null)
  const { startNewConversation } = useNewConversation()
  
  // Use provided callback or fall back to the hook default
  const handleNewConversation = onNewConversation || startNewConversation

  // Use provided conversation select or fall back to navigation route
  const handleConversationSelect = onConversationSelect || ((threadId: string) => {
    router.push(`/c/${threadId}`)
  })

  // Reset navigation state when pathname changes
  useEffect(() => {
    setNavigatingTo(null)
  }, [pathname])
  
  const navItems = [
    { href: "/", label: "Orchestrator", icon: Workflow },
    { href: "/agents", label: "Agent Directory", icon: Users },
    { href: "/saved-workflows", label: "Saved Workflows", icon: Save },
    { href: "/schedules", label: "Schedules", tooltipLabel: "Scheduled Workflows", icon: CalendarClock },
    { href: "/connections", label: "Connections", icon: Link2 },
  ]

  const handleNavClick = (href: string) => {
    if (pathname !== href) {
      setNavigatingTo(href)
    }
  }

  return (
    <>
      {/* Mini Sidebar - Always visible */}
      <TooltipProvider delayDuration={0}>
        <div className="fixed left-0 top-0 h-screen w-16 bg-bg-card border-r border-border-color z-40 flex flex-col items-center py-4 gap-2">
            <Link href="/" className="flex items-center gap-2 bottom-0 mb-4 transition-transform active:scale-95">
              <Image
                src="/logo.png"
                alt="Orbimesh Logo"
                width={32}
                height={32}
                className="rounded"
                priority
              />
            </Link>

            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={handleNewConversation}
                  className="p-2 rounded-orbimesh-lg hover:bg-bg-hover flex items-center justify-center w-10 h-10 active:scale-95 transition-all"
                  aria-label="New conversation"
                >
                  <Plus className="w-5 h-5 text-text-secondary" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>New Conversation</p>
              </TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={toggleSidebar}
                  className={cn(
                    "p-2 rounded-orbimesh-lg hover:bg-bg-hover transition-all flex items-center justify-center w-10 h-10 active:scale-95",
                    open && "bg-bg-hover text-text-primary"
                  )}
                  aria-label="Conversation history"
                >
                  <History className="w-5 h-5 text-text-secondary" />
                </button>
              </TooltipTrigger>
              <TooltipContent side="right">
                <p>Conversation History</p>
              </TooltipContent>
            </Tooltip>
            
            {/* Mini navigation icons */}
            <div className="flex flex-col gap-2 items-center w-full mt-2">
              {navItems.map((item) => {
                const isActive = pathname === item.href
                const isNavigating = navigatingTo === item.href

                return (
                  <Tooltip key={item.href}>
                    <TooltipTrigger asChild>
                      <Link
                        href={item.href}
                        onClick={() => handleNavClick(item.href)}
                        className={cn(
                          "p-2 rounded-orbimesh-lg flex items-center justify-center w-10 h-10 mx-auto transition-all duration-200",
                          "hover:bg-bg-hover active:scale-90",
                          isActive 
                            ? "bg-status-active-light text-status-active-dark shadow-sm ring-1 ring-status-active-border" 
                            : "text-text-secondary hover:text-text-primary",
                          isNavigating && "opacity-80 cursor-wait bg-bg-subtle"
                        )}
                        aria-label={item.label}
                      >
                        {isNavigating ? (
                          <Loader2 className="w-5 h-5 animate-spin text-status-active-dark" />
                        ) : (
                          <item.icon className={cn(
                            "w-5 h-5 transition-colors",
                            isActive ? "text-status-active-dark" : "text-text-secondary" 
                          )} />
                        )}
                      </Link>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                      <p>{item.tooltipLabel ?? item.label}</p>
                    </TooltipContent>
                  </Tooltip>
                )
              })}
            </div>
            <div className="mt-auto flex flex-col items-center gap-2 pb-2">
                <Tooltip>
                    <TooltipTrigger asChild>
                        <div>
                            <ThemeToggle />
                        </div>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                        <p>Toggle Theme</p>
                    </TooltipContent>
                </Tooltip>
              <Tooltip>
                <TooltipTrigger asChild>
                  <Link
                    href="/settings"
                    className="p-2 rounded-orbimesh-lg hover:bg-bg-hover transition-colors flex items-center justify-center w-10 h-10 active:scale-95"
                    aria-label="Settings"
                  >
                    <Settings className="w-5 h-5 text-text-secondary" />
                  </Link>
                </TooltipTrigger>
                <TooltipContent side="right">
                  <p>Settings</p>
                </TooltipContent>
              </Tooltip>
              <UserButtonWrapper />
            </div>
          </div>
      </TooltipProvider>

      {/* Conversation History Sidebar */}
      <div
        className={cn(
          "fixed top-0 left-16 h-screen w-64 bg-bg-card border-r border-border-color z-30 transition-transform duration-200 ease-out",
          open ? "translate-x-0" : "-translate-x-full"
        )}
      >
        <div className="flex h-full flex-col overflow-hidden">
          <SidebarHeader className="border-b border-border-color p-4 flex-shrink-0">
            <div className="flex items-center justify-between">
              <span className="ui-nav-brand">Conversations</span>
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-orbimesh-lg hover:bg-bg-hover transition-colors flex items-center justify-center"
                aria-label="Close conversation history"
              >
                <X className="w-5 h-5 text-text-secondary" />
              </button>
            </div>
          </SidebarHeader>

          <SidebarContent className="flex-1 overflow-y-auto">
            <SidebarGroup>
              <SidebarGroupContent className="px-3 pt-2 flex flex-col">
                <div className="w-full flex flex-col">
                  <ConversationsDropdown
                    onConversationSelect={handleConversationSelect}
                    onNewConversation={handleNewConversation}
                    currentThreadId={currentThreadId}
                  />
                </div>
              </SidebarGroupContent>
            </SidebarGroup>
          </SidebarContent>
        </div>
      </div>
    </>
  )
}

