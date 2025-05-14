import type {
  ApiResponse,
  PaginatedResponse,
  RainfallMetrics,
  WealthMetrics,
  UnifiedDataset,
  DataQuery,
  RegionQuery,
  RainfallDataPoint,
  WealthDataPoint,
  GeoJSON,
  ApiQueryParams,
} from '@/types/data';

class ApiClient {
  private baseUrl: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private cacheExpiry: number;

  constructor(baseUrl: string = '/api', cacheExpiry: number = 5 * 60 * 1000) {
    this.baseUrl = baseUrl;
    this.cache = new Map();
    this.cacheExpiry = cacheExpiry;
  }

  private async fetchWithCache<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const cacheKey = `${endpoint}-${JSON.stringify(options)}`;
    const cached = this.cache.get(cacheKey);

    if (cached && Date.now() - cached.timestamp < this.cacheExpiry) {
      return cached.data as T;
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, options);
    if (!response.ok) {
      throw new Error(`API Error: ${response.statusText}`);
    }

    const data = await response.json();
    this.cache.set(cacheKey, { data, timestamp: Date.now() });
    return data;
  }

  private createQueryString(params: ApiQueryParams): string {
    const urlParams = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined) {
        urlParams.set(key, value);
      }
    });
    const queryString = urlParams.toString();
    return queryString ? `?${queryString}` : '';
  }

  private formatDataQuery(query: DataQuery): ApiQueryParams {
    return {
      ...query,
      dateRange: query.dateRange ? JSON.stringify(query.dateRange) : undefined,
      metrics: query.metrics?.join(','),
    };
  }

  // Region data
  async getRegions(query?: RegionQuery): Promise<ApiResponse<UnifiedDataset[]>> {
    const queryString = this.createQueryString(query || {});
    return this.fetchWithCache(`/regions${queryString}`);
  }

  async getRegionBoundaries(): Promise<ApiResponse<GeoJSON>> {
    return this.fetchWithCache('/regions/boundaries');
  }

  // Rainfall data
  async getRainfallMetrics(query?: RegionQuery): Promise<ApiResponse<RainfallMetrics[]>> {
    const queryString = this.createQueryString(query || {});
    return this.fetchWithCache(`/rainfall/metrics${queryString}`);
  }

  async getRainfallData(query: DataQuery): Promise<PaginatedResponse<RainfallDataPoint[]>> {
    const params = this.formatDataQuery(query);
    const queryString = this.createQueryString(params);
    return this.fetchWithCache(`/rainfall/data${queryString}`);
  }

  // Wealth data
  async getWealthMetrics(query?: RegionQuery): Promise<ApiResponse<WealthMetrics[]>> {
    const queryString = this.createQueryString(query || {});
    return this.fetchWithCache(`/wealth/metrics${queryString}`);
  }

  async getWealthPoints(query: DataQuery): Promise<PaginatedResponse<WealthDataPoint[]>> {
    const params = this.formatDataQuery(query);
    const queryString = this.createQueryString(params);
    return this.fetchWithCache(`/wealth/points${queryString}`);
  }

  // Unified data
  async getUnifiedData(query?: DataQuery): Promise<ApiResponse<UnifiedDataset[]>> {
    const params = query ? this.formatDataQuery(query) : {};
    const queryString = this.createQueryString(params);
    return this.fetchWithCache(`/unified${queryString}`);
  }

  // Cache management
  clearCache(): void {
    this.cache.clear();
  }

  invalidateCache(endpoint: string): void {
    for (const key of this.cache.keys()) {
      if (key.startsWith(endpoint)) {
        this.cache.delete(key);
      }
    }
  }
}

// Export singleton instance
export const api = new ApiClient();

// Export class for testing/custom instances
export default ApiClient; 