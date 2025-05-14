import React from 'react';
import { Link, useLocation } from 'react-router-dom';
import { cn } from '@/lib/utils';

interface NavItem {
  href: string;
  label: string;
  icon?: React.ReactNode;
}

const navItems: NavItem[] = [
  { href: '/', label: 'Overview' },
  { href: '/rainfall', label: 'Rainfall Analysis' },
  { href: '/wealth', label: 'Wealth Analysis' },
  { href: '/regions', label: 'Region Boundaries' },
  { href: '/unified', label: 'Unified Dataset' },
];

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

export function Sidebar({ isOpen, onClose }: SidebarProps) {
  const location = useLocation();

  return (
    <>
      {/* Backdrop for mobile */}
      <div
        className={cn(
          'fixed inset-0 z-20 bg-black/80 lg:hidden',
          isOpen ? 'block' : 'hidden'
        )}
        onClick={onClose}
      />

      {/* Sidebar */}
      <div
        className={cn(
          'fixed inset-y-0 left-0 z-30 w-64 bg-background transform transition-transform duration-200 ease-in-out lg:translate-x-0 lg:static lg:w-auto',
          isOpen ? 'translate-x-0' : '-translate-x-full'
        )}
      >
        <nav className="h-full flex flex-col p-4">
          <div className="space-y-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                to={item.href}
                className={cn(
                  'flex items-center px-4 py-2 text-sm font-medium rounded-md',
                  location.pathname === item.href
                    ? 'bg-primary text-primary-foreground'
                    : 'text-muted-foreground hover:bg-accent hover:text-accent-foreground'
                )}
                onClick={() => {
                  if (window.innerWidth < 1024) {
                    onClose();
                  }
                }}
              >
                {item.icon}
                <span>{item.label}</span>
              </Link>
            ))}
          </div>
        </nav>
      </div>
    </>
  );
} 