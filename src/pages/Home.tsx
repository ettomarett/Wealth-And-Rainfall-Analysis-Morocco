import React from 'react';

export default function Home() {
  return (
    <div className="p-6">
      <h1 className="text-3xl font-bold mb-6">Morocco Data Analysis Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <div className="p-6 rounded-lg border border-border bg-card text-card-foreground shadow">
          <h2 className="text-xl font-semibold mb-2">Rainfall Analysis</h2>
          <p className="text-muted-foreground">Analyze rainfall patterns across different regions of Morocco.</p>
        </div>
        <div className="p-6 rounded-lg border border-border bg-card text-card-foreground shadow">
          <h2 className="text-xl font-semibold mb-2">Wealth Distribution</h2>
          <p className="text-muted-foreground">Explore wealth distribution patterns and economic indicators.</p>
        </div>
        <div className="p-6 rounded-lg border border-border bg-card text-card-foreground shadow">
          <h2 className="text-xl font-semibold mb-2">Region Boundaries</h2>
          <p className="text-muted-foreground">View and analyze administrative boundaries and regional data.</p>
        </div>
      </div>
    </div>
  );
} 