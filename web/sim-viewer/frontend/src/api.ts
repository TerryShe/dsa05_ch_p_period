// Types and fetchers of the backend API (server/app.py).

export interface Listing {
  id: string;
  title: string;
  report: string;
  shown: boolean;
}

export interface Group {
  name: string;
  description: string;
  grid: number[];
  layers: number;
}

export interface ComparisonRow {
  key: string;
  label: string;
  difference: number;
  ratio: number | null;
  ci95: [number, number];
  deployments: number | null;
}

export interface Family {
  id: string;
  title: string;
  note?: string;
  scale: "rate" | "log";
  core?: string[];
  rows: ComparisonRow[];
}

export type ByGroupAndN = Record<string, Record<string, number>>;

export interface Overview {
  id: string;
  title: string;
  report: string;
  evaluation_set: string;
  instances: number;
  commits: string[];
  groups: Group[];
  decision: string;
  tables: {
    P2: ByGroupAndN;
    P1: ByGroupAndN;
    p90_w: ByGroupAndN;
    all_ue_mean_rate: ByGroupAndN;
    N_star_on_grid: Record<string, number | null>;
  };
  comparisons: Family[];
}

export interface Side {
  group: string;
  layers: number;
  w: number[];
  P2: number;
  P2_analysis: number;
  P1: number;
}

export interface Paired {
  n: number;
  instances: number[];
  a: Side;
  b: Side;
}

export interface Layer {
  spec: string;
  violation_rate: number[];
  violations: number[];
  undeliverable: number[];
  w: number;
}

export interface Deployment {
  instance: number;
  n: number;
  records: {
    packets: number[];
    class_index: number[];
    distance_m: number[];
    shadowing_db: number[];
  };
  groups: Record<string, { layers: Layer[]; rate_mean: number[]; w: number }>;
  geometry: {
    radius_m: number;
    positions_m: [number, number][];
    distance_m: number[];
    shadowing_db: number[];
    class_names: string[];
    class_index: number[];
    period_slots: number[];
    packet_bits: number[];
  };
}

async function get<T>(path: string, params: Record<string, string | number> = {}): Promise<T> {
  const query = new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]));
  const response = await fetch(query.size ? `${path}?${query}` : path);
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const api = {
  experiments: () => get<Listing[]>("/api/experiments"),
  overview: (id: string) => get<Overview>(`/api/experiments/${id}`),
  paired: (id: string, a: string, b: string, n: number) =>
    get<Paired>(`/api/experiments/${id}/paired`, { a, b, n }),
  deployment: (id: string, instance: number, n: number, groups: string[]) =>
    get<Deployment>(`/api/experiments/${id}/deployments/${instance}`, {
      n,
      groups: [...new Set(groups)].join(","),
    }),
};
