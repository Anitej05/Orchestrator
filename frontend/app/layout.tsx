import type React from "react"
import type { Metadata } from "next"
import { Inter } from "next/font/google"
import { ClerkProvider } from '@clerk/nextjs'
import { SidebarProvider } from "@/components/ui/sidebar"
import AppSidebar from "@/components/app-sidebar"
import "@/styles/globals.css"
import { Toaster } from "@/components/ui/toaster"
import { ThemeProvider } from "@/components/theme-toggle"

const inter = Inter({ subsets: ["latin"] })

export const metadata: Metadata = {
  title: "Orbimesh",
  description: "Discover, test, and deploy AI agents for your business workflows",
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <ClerkProvider>
      <html lang="en" suppressHydrationWarning className="h-screen overflow-hidden">
        <body className={`${inter.className} h-screen overflow-hidden`} suppressHydrationWarning>
          <ThemeProvider
            attribute="class"
            defaultTheme="light"
            enableSystem={false}
            storageKey="orbimesh-theme"
            disableTransitionOnChange
          >
            <SidebarProvider defaultOpen={false}>
              <AppSidebar />
              <div className="ml-16 min-h-screen w-full max-w-full overflow-hidden">
                {children}
              </div>
              <Toaster />
            </SidebarProvider>
          </ThemeProvider>
        </body>
      </html>
    </ClerkProvider>
  )
}

