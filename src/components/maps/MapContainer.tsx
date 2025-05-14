import { BaseMap, BaseMapProps } from './BaseMap';

export interface MapContainerProps extends BaseMapProps {
  height?: string;
}

export function MapContainer({ height = '600px', className = '', ...props }: MapContainerProps) {
  return (
    <div style={{ height }} className={`rounded-lg border border-border bg-card ${className}`}>
      <BaseMap {...props} className="rounded-lg" />
    </div>
  );
} 