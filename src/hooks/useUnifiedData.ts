import { useQuery } from '@tanstack/react-query';
import { api } from '@/lib/api/client';
import type { DataQuery } from '@/types/data';

export function useUnifiedData(query?: DataQuery) {
  return useQuery({
    queryKey: ['unifiedData', query],
    queryFn: () => api.getUnifiedData(query),
  });
} 