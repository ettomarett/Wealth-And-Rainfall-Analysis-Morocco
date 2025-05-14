// Base types
export interface Region {
  id: string;
  name: string;
  province: string;
  population: number;
  area_km2: number;
}

export interface GeoFeature {
  type: 'Feature';
  geometry: {
    type: string;
    coordinates: number[][] | number[][][] | number[][][][];
  };
  properties: Record<string, any>;
}

export interface GeoJSON {
  type: 'FeatureCollection';
  features: GeoFeature[];
}

// Rainfall data
export interface RainfallMetrics {
  region_id: string;
  annual_mean: number;
  seasonal_variation: number;
  max_monthly: number;
  min_monthly: number;
  drought_months: number;
}

export interface RainfallDataPoint {
  region_id: string;
  date: string;
  rainfall_mm: number;
}

// Wealth data
export interface WealthMetrics {
  region_id: string;
  mean_rwi: number;
  median_rwi: number;
  std_rwi: number;
  points_count: number;
}

export interface WealthDataPoint {
  latitude: number;
  longitude: number;
  rwi: number;
  region_id?: string;
}

// Unified data
export interface UnifiedDataset extends Region {
  rainfall_metrics: RainfallMetrics;
  wealth_metrics: WealthMetrics;
  geometry?: GeoFeature['geometry'];
}

// API Response types
export interface ApiResponse<T> {
  data: T;
  error?: string;
}

export interface PaginatedResponse<T> extends ApiResponse<T> {
  total: number;
  page: number;
  pageSize: number;
}

// Query parameters
export interface DateRange {
  start: string;
  end: string;
}

export interface RegionQuery {
  [key: string]: string | undefined;
  regionId?: string;
  province?: string;
}

export interface DataQuery extends RegionQuery {
  dateRange?: DateRange;
  metrics?: string[];
}

// Utility type for API parameters
export type ApiQueryParams = {
  [key: string]: string | undefined;
};

export interface RainfallData {
  region_id: string;
  region_name: string;
  year: number;
  month: number;
  rainfall_mm: number;
}

export interface WealthData {
  region_id: string;
  region_name: string;
  rwi_score: number;
  latitude: number;
  longitude: number;
}

export interface RegionBoundary {
  id: string;
  name: string;
  geometry: GeoJSON.Geometry;
  properties: {
    admin_level: number;
    parent_region?: string;
  };
}

export interface UnifiedDataPoint {
  region_id: string;
  region_name: string;
  year: number;
  month: number;
  rainfall_mm: number;
  rwi_score: number;
  geometry: GeoJSON.Geometry;
}

export interface DatasetMetadata {
  lastUpdated: string;
  totalRegions: number;
  timeRange: {
    start: string;
    end: string;
  };
  dataPoints: number;
} 