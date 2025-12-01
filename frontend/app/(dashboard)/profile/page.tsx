"use client"

import { UserProfile } from "@clerk/nextjs"
import AppSidebar from "@/components/app-sidebar"
import Navbar from "@/components/navbar"
import { SidebarProvider, SidebarInset, SidebarTrigger, useSidebar } from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

function ProfileContent() {
  const { open } = useSidebar()
  
  return (
    <SidebarInset className={!open ? "ml-16" : ""}>
          <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800">
            {/* Header */}
            <div className="bg-white/80 dark:bg-gray-800/80 backdrop-blur-sm border-b dark:border-gray-700 px-6 py-4">
              <div className="flex items-center space-x-4">
                <SidebarTrigger />
              </div>
            </div>

            {/* Main Content */}
            <main className="p-6">
              {/* Title Section */}
              <div className="mb-6">
                <h1 className="text-3xl font-bold text-blue-600 dark:text-blue-400">Profile & Settings</h1>
                <p className="text-gray-600 dark:text-gray-300 mt-2">
                  Manage your account settings and preferences
                </p>
              </div>

              <Tabs defaultValue="profile" className="w-full">
                <TabsList className="mb-6">
                  <TabsTrigger value="profile">Profile</TabsTrigger>
                  <TabsTrigger value="preferences">Preferences</TabsTrigger>
                </TabsList>

                <TabsContent value="profile">
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Account Profile</CardTitle>
                      <CardDescription>
                        Manage your personal information, email, and security settings
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="flex justify-center">
                      <UserProfile 
                        routing="hash"
                        appearance={{
                          elements: {
                            rootBox: "w-full",
                            card: "shadow-none border-0",
                          }
                        }}
                      />
                    </CardContent>
                  </Card>
                </TabsContent>

                <TabsContent value="preferences">
                  <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700">
                    <CardHeader>
                      <CardTitle>Application Preferences</CardTitle>
                      <CardDescription>
                        Customize your Orbimesh experience
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="space-y-4">
                        <div>
                          <h3 className="text-lg font-medium mb-2">Theme</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Theme preferences are managed through the theme toggle in the navigation bar.
                          </p>
                        </div>
                        
                        <div>
                          <h3 className="text-lg font-medium mb-2">Notifications</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            Notification settings will be available in a future update.
                          </p>
                        </div>

                        <div>
                          <h3 className="text-lg font-medium mb-2">API Access</h3>
                          <p className="text-sm text-gray-600 dark:text-gray-400">
                            API keys and webhook management will be available in a future update.
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </TabsContent>
              </Tabs>
            </main>
          </div>
        </SidebarInset>
  )
}

export default function ProfilePage() {
  return (
    <>
      <Navbar />
      <SidebarProvider>
        <AppSidebar />
        <ProfileContent />
      </SidebarProvider>
    </>
  )
}
