import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api/client';
import type { RegionQuery, DataQuery } from '@/types/data';

export function useRainfallMetrics(query?: RegionQuery) {
  return useQuery({
    queryKey: ['rainfallMetrics', query],
    queryFn: () => api.getRainfallMetrics(query),
  });
}

export function useRainfallData(query: DataQuery) {
  return useQuery({
    queryKey: ['rainfallData', query],
    queryFn: () => api.getRainfallData(query),
  });
} 