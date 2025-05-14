import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import { useState } from 'react';
import Home from './pages/Home';
import RainfallAnalysis from './pages/RainfallAnalysis';
import WealthAnalysis from './pages/WealthAnalysis';
import RegionBoundaries from './pages/RegionBoundaries';
import UnifiedDataset from './pages/UnifiedDataset';
import { Sidebar } from './components/layout/Sidebar';
import { ThemeProvider } from './components/theme/ThemeProvider';
import { ThemeToggle } from './components/theme/ThemeToggle';
import { QueryProvider } from './providers/QueryProvider';
import './styles/globals.css';

function App() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <QueryProvider>
      <ThemeProvider defaultTheme="system" storageKey="dddm-theme">
        <Router>
          <div className="min-h-screen bg-background">
            {/* Navigation */}
            <nav className="border-b border-border">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between h-16">
                  <div className="flex">
                    {/* Mobile menu button */}
                    <button
                      type="button"
                      className="lg:hidden inline-flex items-center justify-center p-2 rounded-md text-foreground"
                      onClick={() => setIsSidebarOpen(true)}
                    >
                      <span className="sr-only">Open menu</span>
                      <svg
                        className="h-6 w-6"
                        fill="none"
                        viewBox="0 0 24 24"
                        strokeWidth="1.5"
                        stroke="currentColor"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5"
                        />
                      </svg>
                    </button>

                    <Link to="/" className="flex items-center px-2 text-xl font-bold">
                      DDDM
                    </Link>
                    <div className="hidden sm:ml-6 sm:flex sm:space-x-8">
                      <Link to="/rainfall" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-foreground">
                        Rainfall
                      </Link>
                      <Link to="/wealth" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-foreground">
                        Wealth
                      </Link>
                      <Link to="/regions" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-foreground">
                        Regions
                      </Link>
                      <Link to="/unified" className="inline-flex items-center px-1 pt-1 text-sm font-medium text-foreground">
                        Unified Data
                      </Link>
                    </div>
                  </div>

                  {/* Theme toggle */}
                  <div className="flex items-center">
                    <ThemeToggle />
                  </div>
                </div>
              </div>
            </nav>

            {/* Sidebar */}
            <Sidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} />

            {/* Main content */}
            <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
              <Routes>
                <Route path="/" element={<Home />} />
                <Route path="/rainfall" element={<RainfallAnalysis />} />
                <Route path="/wealth" element={<WealthAnalysis />} />
                <Route path="/regions" element={<RegionBoundaries />} />
                <Route path="/unified" element={<UnifiedDataset />} />
              </Routes>
            </main>
          </div>
        </Router>
      </ThemeProvider>
    </QueryProvider>
  );
}

export default App; 