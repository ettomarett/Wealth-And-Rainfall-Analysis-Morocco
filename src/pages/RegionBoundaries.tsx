import { useRegionBoundaries } from '@/hooks';
import { MapContainer } from '@/components/maps/MapContainer';
import { Layer, Source } from 'react-map-gl';

export default function RegionBoundaries() {
  const { data: boundaries, isLoading, error } = useRegionBoundaries();

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Region Boundaries</h1>
        <p className="text-muted-foreground">
          View and explore administrative boundaries across regions.
        </p>
      </div>

      <div className="grid gap-6">
        {/* Map section */}
        <MapContainer>
          {boundaries && (
            <Source id="region-boundaries" type="geojson" data={boundaries.data}>
              <Layer
                id="region-boundaries-fill"
                type="fill"
                paint={{
                  'fill-color': '#088',
                  'fill-opacity': 0.2,
                }}
              />
              <Layer
                id="region-boundaries-line"
                type="line"
                paint={{
                  'line-color': '#088',
                  'line-width': 1,
                }}
              />
            </Source>
          )}
        </MapContainer>

        {/* Loading and error states */}
        {isLoading && (
          <div className="flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
          </div>
        )}
        {error && (
          <div className="rounded-lg bg-destructive/10 p-4 text-destructive">
            Error loading region boundaries
          </div>
        )}
      </div>
    </div>
  );
} 