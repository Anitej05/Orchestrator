'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import Navbar from '@/components/navbar';
import { Home, Workflow, Calendar } from 'lucide-react';

export default function NotFound() {
  const router = useRouter();

  return (
    <>
      <Navbar />
      <div className="min-h-screen bg-gradient-to-br from-gray-50 to-blue-50 dark:from-gray-900 dark:to-gray-800 pt-20">
        <div className="flex items-center justify-center min-h-[calc(100vh-5rem)]">
          <div className="text-center px-4">
            <div className="mb-8">
              <h1 className="text-6xl font-bold text-gray-900 dark:text-white mb-4">404</h1>
              <h2 className="text-3xl font-semibold text-gray-800 dark:text-gray-100 mb-2">Page Not Found</h2>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8">
                The page you're looking for doesn't exist or has been moved.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 max-w-2xl mx-auto">
              <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => router.push('/')}>
                <CardContent className="p-6 text-center">
                  <Home className="w-8 h-8 mx-auto mb-3 text-blue-600 dark:text-blue-400" />
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Home</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Go to the homepage</p>
                </CardContent>
              </Card>

              <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => router.push('/workflows')}>
                <CardContent className="p-6 text-center">
                  <Workflow className="w-8 h-8 mx-auto mb-3 text-blue-600 dark:text-blue-400" />
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Workflows</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">Browse your workflows</p>
                </CardContent>
              </Card>

              <Card className="bg-white dark:bg-gray-800 border-gray-200 dark:border-gray-700 hover:shadow-lg transition-shadow cursor-pointer" onClick={() => router.push('/schedules')}>
                <CardContent className="p-6 text-center">
                  <Calendar className="w-8 h-8 mx-auto mb-3 text-blue-600 dark:text-blue-400" />
                  <h3 className="font-semibold text-gray-900 dark:text-white mb-1">Schedules</h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">View scheduled workflows</p>
                </CardContent>
              </Card>
            </div>

            <div className="space-x-4">
              <Button
                onClick={() => router.back()}
                variant="outline"
                className="border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                Go Back
              </Button>
              <Button
                onClick={() => router.push('/')}
                className="bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800"
              >
                Return Home
              </Button>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}