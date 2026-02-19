'use client';

import { useRouter } from 'next/navigation';
import { Button } from '@/components/ui/button';
import { Card, CardContent } from '@/components/ui/card';
import { Home, Workflow, Calendar } from 'lucide-react';

export default function NotFound() {
  const router = useRouter();

  return (
      <div className="min-h-screen bg-bg-page flex items-center justify-center">
        <div className="w-full max-w-3xl text-center px-6">
            <div className="mb-8">
              <h1 className="text-6xl font-bold text-text-primary mb-4">404</h1>
              <h2 className="text-3xl font-semibold text-text-primary mb-2">Page Not Found</h2>
              <p className="text-lg text-text-secondary mb-8">
                The page you're looking for doesn't exist or has been moved.
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 max-w-2xl mx-auto">
              <Card className="ui-card hover:shadow-orbimesh-card-hover transition-shadow cursor-pointer" onClick={() => router.push('/')}>
                <CardContent className="p-6 text-center">
                  <Home className="w-8 h-8 mx-auto mb-3 text-brand-teal" />
                  <h3 className="font-semibold text-text-primary mb-1">Home</h3>
                  <p className="text-sm text-text-secondary">Go to the homepage</p>
                </CardContent>
              </Card>

              <Card className="ui-card hover:shadow-orbimesh-card-hover transition-shadow cursor-pointer" onClick={() => router.push('/saved-workflows')}>
                <CardContent className="p-6 text-center">
                  <Workflow className="w-8 h-8 mx-auto mb-3 text-brand-teal" />
                  <h3 className="font-semibold text-text-primary mb-1">Workflows</h3>
                  <p className="text-sm text-text-secondary">Browse your workflows</p>
                </CardContent>
              </Card>

              <Card className="ui-card hover:shadow-orbimesh-card-hover transition-shadow cursor-pointer" onClick={() => router.push('/schedules')}>
                <CardContent className="p-6 text-center">
                  <Calendar className="w-8 h-8 mx-auto mb-3 text-brand-teal" />
                  <h3 className="font-semibold text-text-primary mb-1">Schedules</h3>
                  <p className="text-sm text-text-secondary">View scheduled workflows</p>
                </CardContent>
              </Card>
            </div>

            <div className="space-x-4">
              <Button
                onClick={() => router.back()}
                variant="outline"
                className="border-border-color text-text-secondary hover:bg-bg-hover"
              >
                Go Back
              </Button>
              <Button
                onClick={() => router.push('/')}
                className="bg-brand-teal hover:bg-brand-teal-hover text-white"
              >
                Return Home
              </Button>
            </div>
        </div>
      </div>
  );
}
