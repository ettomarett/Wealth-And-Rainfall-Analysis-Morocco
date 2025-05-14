export default function Home() {
  return (
    <div className="space-y-8">
      <section className="space-y-4">
        <h1 className="text-4xl font-bold">DDDM Analytics Dashboard</h1>
        <p className="text-lg text-muted-foreground">
          Explore Morocco's regional data through interactive visualizations and analysis tools.
        </p>
      </section>
      
      <section className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg border bg-card p-4">
          <h2 className="font-semibold">Rainfall Analysis</h2>
          <p className="text-sm text-muted-foreground">
            Analyze rainfall patterns and trends across different regions.
          </p>
        </div>
        
        <div className="rounded-lg border bg-card p-4">
          <h2 className="font-semibold">Wealth Analysis</h2>
          <p className="text-sm text-muted-foreground">
            Explore wealth distribution and economic indicators.
          </p>
        </div>
        
        <div className="rounded-lg border bg-card p-4">
          <h2 className="font-semibold">Region Boundaries</h2>
          <p className="text-sm text-muted-foreground">
            Visualize administrative boundaries and regional data.
          </p>
        </div>
        
        <div className="rounded-lg border bg-card p-4">
          <h2 className="font-semibold">Unified Dataset</h2>
          <p className="text-sm text-muted-foreground">
            Access combined analysis of rainfall and wealth data.
          </p>
        </div>
      </section>
    </div>
  );
} 