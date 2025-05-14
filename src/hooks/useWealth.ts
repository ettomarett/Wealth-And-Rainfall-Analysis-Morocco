import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api/client';
import type { RegionQuery, DataQuery } from '@/types/data';

export function useWealthMetrics(query?: RegionQuery) {
  return useQuery({
    queryKey: ['wealthMetrics', query],
    queryFn: () => api.getWealthMetrics(query),
  });
}

export function useWealthPoints(query: DataQuery) {
  return useQuery({
    queryKey: ['wealthPoints', query],
    queryFn: () => api.getWealthPoints(query),
  });
} 