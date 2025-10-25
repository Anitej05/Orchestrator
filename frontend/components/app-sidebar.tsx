// components/app-sidebar.tsx
"use client"

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
import ConversationsDropdown from "./conversations-dropdown"

interface AppSidebarProps {
  onConversationSelect?: (threadId: string) => void;
  currentThreadId?: string;
}

export default function AppSidebar({ onConversationSelect, currentThreadId }: AppSidebarProps) {
  const navItems = [
    { href: "/", label: "Orchestrator", icon: Workflow },
    { href: "/agents", label: "Agent Directory", icon: Users },
    { href: "/metrics", label: "Metrics & Dashboard", icon: BarChart3 },
    { href: "/profile", label: "Profile / Settings", icon: Settings },
  ]

  return (
    <Sidebar>
      <SidebarHeader className="border-b border-gray-200 p-4">
        <Link href="/" className="flex items-center space-x-2">
          <Bot className="w-8 h-8 text-blue-600" />
          <span className="text-xl font-bold text-blue-600">Orbimesh</span>
        </Link>
      </SidebarHeader>

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
                currentThreadId={currentThreadId}
              />
            )}
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>

      <SidebarFooter className="border-t border-gray-200 p-4">
        <Link href="/register-agent">
          <Button className="w-full bg-blue-600 hover:bg-blue-700 text-white">
            <Plus className="w-4 h-4 mr-2" />
            Register Agent
          </Button>
        </Link>
      </SidebarFooter>
    </Sidebar>
  )
}
