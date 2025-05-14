import { useCallback, useState } from 'react';
import Map, { MapRef, ViewStateChangeEvent } from 'react-map-gl';
import type { ViewState } from 'react-map-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

export interface BaseMapProps {
  initialViewState?: Partial<ViewState>;
  children?: React.ReactNode;
  onMapLoad?: (map: MapRef) => void;
  onMapMove?: (viewState: ViewState) => void;
  className?: string;
}

const DEFAULT_VIEW_STATE: Partial<ViewState> = {
  longitude: 120.9842, // Philippines center
  latitude: 14.5995,
  zoom: 5,
  pitch: 0,
  bearing: 0,
};

export function BaseMap({
  initialViewState = DEFAULT_VIEW_STATE,
  children,
  onMapLoad,
  onMapMove,
  className = '',
}: BaseMapProps) {
  const [viewState, setViewState] = useState(initialViewState);
  const mapRef = useCallback((ref: MapRef | null) => {
    if (ref && onMapLoad) {
      onMapLoad(ref);
    }
  }, [onMapLoad]);

  return (
    <Map
      ref={mapRef}
      {...viewState}
      onMove={(evt: ViewStateChangeEvent) => {
        setViewState(evt.viewState);
        onMapMove?.(evt.viewState);
      }}
      style={{ width: '100%', height: '100%' }}
      className={className}
      mapStyle="mapbox://styles/mapbox/light-v11"
      mapboxAccessToken={import.meta.env.VITE_MAPBOX_TOKEN}
    >
      {children}
    </Map>
  );
} 