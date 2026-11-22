// JSON shapes returned by the FastAPI backend (api/main.py, LLD section 9). Keep in sync with the backend.

export type Health = {
  status: string;
  detail?: string | null;
  systems?: string[];
  methods?: string[];
  loaded?: Record<string, boolean>;
};

export type QueryTerm = { term: string; df: number; idf: number; weight: number };

export type Breakdown = {
  bm25: number;
  neural: number | null;
  fusion: number | null;
  pagerank_citation: number | null;
  pagerank_similarity: number | null;
};

export type ResultItem = {
  rank: number;
  doc_id: string;
  title: string;
  snippet: string;
  matched_passages: string[];
  is_relevant_in_benchmark: boolean | null;
  breakdown: Breakdown;
};

export type SearchResponse = {
  system: string;
  known_query_id: string | null;
  query_terms: QueryTerm[];
  num_query_terms: number;
  results: ResultItem[];
  timings_ms: Record<string, number>;
};

export type StatuteRow = {
  id: string;
  title: string;
  score: number;
  threshold: number;
  probability: number | null;
  above_threshold: boolean;
};

export type StatuteResponse = { method: string; statutes: StatuteRow[] };

export type Example = { id: string; title: string; text: string; num_relevant: number };

// One row of a long-format results CSV: task, system, split, metric, value, n
export type MetricRow = { task: string; system: string; split: string; metric: string; value: string; n: string };

export type PublishedBaseline = {
  task: string;
  system: string;
  metric: string;
  value_percent: number;
  k: number | null;
  source: string;
};

export type ResultsPayload = {
  pcr_metrics?: MetricRow[];
  lsi_metrics?: MetricRow[];
  efficiency?: MetricRow[];
  significance?: Record<string, string>[];
  published_baselines?: { baselines: PublishedBaseline[]; pcr_metric_definition: string };
  data_stats?: Record<string, unknown>;
  graph_stats?: Record<string, unknown>;
  figures?: string[];
};
