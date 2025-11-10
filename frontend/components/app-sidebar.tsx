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
  useSidebar,
} from "@/components/ui/sidebar"
import { Plus, Home, Users, Workflow, BarChart3, Settings, Bot, Menu, X } from "lucide-react"
import { UserButton } from "@clerk/nextjs"
import ConversationsDropdown from "./conversations-dropdown"

interface AppSidebarProps {
  onConversationSelect?: (threadId: string) => void;
  onNewConversation?: () => void;
  currentThreadId?: string;
}

export default function AppSidebar({ onConversationSelect, onNewConversation, currentThreadId }: AppSidebarProps) {
  const { open, setOpen, toggleSidebar } = useSidebar()
  
  const navItems = [
    { href: "/", label: "Orchestrator", icon: Workflow },
    { href: "/agents", label: "Agent Directory", icon: Users },
    { href: "/metrics", label: "Metrics & Dashboard", icon: BarChart3 },
    { href: "/profile", label: "Profile / Settings", icon: Settings },
  ]

  return (
    <>
      {/* Mini Sidebar - Always visible when sidebar is collapsed */}
      {!open && (
        <div className="fixed left-0 top-[64px] h-[calc(100vh-64px)] w-16 bg-white dark:bg-gray-900 border-r border-gray-200 dark:border-gray-800 z-40 flex flex-col items-center py-4 gap-2">
          <button
            onClick={toggleSidebar}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center justify-center w-10 h-10"
            aria-label="Open sidebar"
          >
            <Menu className="w-6 h-6 text-gray-700 dark:text-gray-300" />
          </button>
          {/* Mini navigation icons */}
          <div className="flex flex-col gap-2 items-center w-full mt-2">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center justify-center w-10 h-10 mx-auto"
                title={item.label}
              >
                <item.icon className="w-5 h-5 text-gray-700 dark:text-gray-300" />
              </Link>
            ))}
          </div>
        </div>
      )}

      {/* Main Sidebar */}
      <Sidebar 
        collapsible="offcanvas" 
        className="border-r z-40 fixed left-0 top-0 h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-950 dark:to-gray-900 [&_[data-sidebar=sidebar]]:bg-transparent [&_[data-sidebar=sidebar]]:bg-gradient-to-br [&_[data-sidebar=sidebar]]:from-gray-50 [&_[data-sidebar=sidebar]]:to-gray-100 [&_[data-sidebar=sidebar]]:dark:from-gray-950 [&_[data-sidebar=sidebar]]:dark:to-gray-900" 
        open={open} 
        onOpenChange={setOpen}
      >
        <div className="flex h-full flex-col pt-16">
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
          {/* <SidebarGroupLabel>Conversations</SidebarGroupLabel> */}
          <SidebarGroupContent className="px-2 pt-2 flex flex-col items-center">
            {onConversationSelect && (
              <div className="w-full flex flex-col items-center">
                <ConversationsDropdown
                  onConversationSelect={onConversationSelect}
                  onNewConversation={onNewConversation}
                  currentThreadId={currentThreadId}
                />
              </div>
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-gray-200 p-4">
        {/* Close button */}
        <button
          onClick={toggleSidebar}
          className="mb-3 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors flex items-center gap-2 w-full"
        >
          <X className="w-5 h-5 text-gray-700 dark:text-gray-300" />
          <span className="text-sm text-gray-700 dark:text-gray-300">Close Sidebar</span>
        </button>
        
        {/* <div className="flex items-center justify-between gap-2 mb-2">
          <UserButton afterSignOutUrl="/sign-in" />
        </div> */}
        <Link href="/register-agent">
          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
            <Plus className="w-4 h-4 mr-2" />
            Register Agent
          </Button>
        </Link>
      </SidebarFooter>
      </div>
    </Sidebar>
    </>
  )
}
