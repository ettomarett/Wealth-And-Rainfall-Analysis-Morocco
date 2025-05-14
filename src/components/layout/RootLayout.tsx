import Link from 'next/link';
import { ReactNode } from 'react';

interface RootLayoutProps {
  children: ReactNode;
}

export function RootLayout({ children }: RootLayoutProps) {
  return (
    <div className="min-h-screen bg-background">
      <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
        <div className="container mx-auto flex h-14 items-center px-4">
          <div className="mr-4 flex">
            <Link href="/" className="mr-6 flex items-center space-x-2">
              <span className="font-bold">DDDM Analytics</span>
            </Link>
          </div>
          <nav className="flex items-center space-x-6 text-sm font-medium">
            <Link href="/rainfall" className="transition-colors hover:text-foreground/80">
              Rainfall Analysis
            </Link>
            <Link href="/wealth" className="transition-colors hover:text-foreground/80">
              Wealth Analysis
            </Link>
            <Link href="/regions" className="transition-colors hover:text-foreground/80">
              Region Boundaries
            </Link>
            <Link href="/unified" className="transition-colors hover:text-foreground/80">
              Unified Dataset
            </Link>
          </nav>
        </div>
      </header>
      <main className="container mx-auto px-4 py-6">
        {children}
      </main>
    </div>
  );
} 