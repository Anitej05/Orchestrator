"use client"

import { UserProfile } from "@clerk/nextjs"
import { SidebarInset, useSidebar } from "@/components/ui/sidebar"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"

function ProfileContent() {
  const { open } = useSidebar()
  
  return (
    <SidebarInset>
      <div className="min-h-screen bg-bg-page dark:bg-background text-text-primary">
        {/* Main Content */}
        <main className="p-6">
          {/* Title Section */}
          <div className="mb-6">
            <h1 className="text-3xl font-bold text-brand-teal">Profile & Settings</h1>
            <p className="text-text-secondary mt-2">
              Manage your account settings and preferences
            </p>
          </div>

              <Tabs defaultValue="profile" className="w-full">
                <TabsList className="mb-6">
                  <TabsTrigger value="profile">Profile</TabsTrigger>
                  <TabsTrigger value="preferences">Preferences</TabsTrigger>
                </TabsList>

                <TabsContent value="profile">
                  <Card className="ui-card">
                    <CardHeader>
                      <CardTitle className="text-text-primary">Account Profile</CardTitle>
                      <CardDescription className="text-text-secondary">
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
                  <Card className="ui-card">
                    <CardHeader>
                      <CardTitle className="text-text-primary">Application Preferences</CardTitle>
                      <CardDescription className="text-text-secondary">
                        Customize your Orbimesh experience
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-6">
                      <div className="space-y-4">
                        <div>
                          <h3 className="text-lg font-medium text-text-primary mb-2">Theme</h3>
                          <p className="text-sm text-text-secondary">
                            Theme preferences are managed through the theme toggle in the navigation bar.
                          </p>
                        </div>
                        
                        <div>
                          <h3 className="text-lg font-medium text-text-primary mb-2">Notifications</h3>
                          <p className="text-sm text-text-secondary">
                            Notification settings will be available in a future update.
                          </p>
                        </div>

                        <div>
                          <h3 className="text-lg font-medium text-text-primary mb-2">API Access</h3>
                          <p className="text-sm text-text-secondary">
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
  return <ProfileContent />
}

