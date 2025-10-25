// components/app-sidebar.tsx
"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import {
  Sidebar,
  SidebarContent,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarFooter,
} from "@/components/ui/sidebar"
import { Plus, Home, Users, Workflow, BarChart3, Settings, Bot } from "lucide-react"
import { UserButton } from "@clerk/nextjs"
import ConversationsDropdown from "./conversations-dropdown"

interface AppSidebarProps {
  onConversationSelect?: (threadId: string) => void;
  onNewConversation?: () => void;
  currentThreadId?: string;
}

export default function AppSidebar({ onConversationSelect, onNewConversation, currentThreadId }: AppSidebarProps) {
  const navItems = [
    { href: "/", label: "Orchestrator", icon: Workflow },
    { href: "/agents", label: "Agent Directory", icon: Users },
    { href: "/metrics", label: "Metrics & Dashboard", icon: BarChart3 },
    { href: "/profile", label: "Profile / Settings", icon: Settings },
  ]

  const [collapsed, setCollapsed] = useState(false);
  return (
    <div className="relative">
      <button
        className="absolute top-2 left-2 z-50 bg-gray-200 rounded-full p-2 shadow hover:bg-gray-300 transition md:hidden"
        onClick={() => setCollapsed((c) => !c)}
        aria-label={collapsed ? "Open sidebar" : "Close sidebar"}
      >
        {collapsed ? (
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-panel-left"><rect width="18" height="18" x="3" y="3" rx="2"/><path d="M9 3v18"/></svg>
        ) : (
          <svg width="24" height="24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="lucide lucide-x"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>
        )}
      </button>
      <Sidebar
        className={`mt-[64px] h-[calc(100vh-64px)] fixed left-0 top-[64px] z-40 transition-all duration-200 ${collapsed ? 'w-0 overflow-hidden' : 'w-64'} md:w-64`}
        collapsible="offcanvas"
        variant="sidebar"
      >
      {/* <SidebarHeader className="border-b border-gray-200 p-4">
        <Link href="/" className="flex items-center space-x-2">
          <Bot className="w-8 h-8 text-blue-600" />
          <span className="text-xl font-bold text-blue-600">Orbimesh</span>
        </Link>
      </SidebarHeader> */}

      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Navigation</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              {navItems.map((item) => (
                <SidebarMenuItem key={item.href + item.label}>
                  <SidebarMenuButton asChild>
                    <Link href={item.href} className="flex items-center space-x-2">
                      <item.icon className="w-4 h-4" />
                      <span>{item.label}</span>
                    </Link>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>

        <SidebarGroup>
          <SidebarGroupLabel>Conversations</SidebarGroupLabel>
          <SidebarGroupContent className="px-2">
            {onConversationSelect && (
              <ConversationsDropdown
                onConversationSelect={onConversationSelect}
                onNewConversation={onNewConversation}
                currentThreadId={currentThreadId}
              />
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-gray-200 p-4">
        <div className="flex items-center justify-between gap-2 mb-2">
          <UserButton afterSignOutUrl="/sign-in" />
        </div>
        <Link href="/register-agent">
          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
            <Plus className="w-4 h-4 mr-2" />
            Register Agent
          </Button>
        </Link>
      </SidebarFooter>
      </Sidebar>
    </div>
  )
}
