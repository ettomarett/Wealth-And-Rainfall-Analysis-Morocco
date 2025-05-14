import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api/client';
import type { RegionQuery, UnifiedDataset } from '@/types/data';

export function useRegions(query?: RegionQuery) {
  return useQuery({
    queryKey: ['regions', query],
    queryFn: () => api.getRegions(query),
  });
}

export function useRegionBoundaries() {
  return useQuery({
    queryKey: ['regionBoundaries'],
    queryFn: () => api.getRegionBoundaries(),
  });
} 