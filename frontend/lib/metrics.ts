// Look up numbers in the results rows returned by /results. Display helpers only: no scoring or ranking here.
import type { MetricRow } from "./types";

export function metricValue(rows: MetricRow[] | undefined, system: string, split: string, metric: string): number | null {
  const row = rows?.find((r) => r.system === system && r.split === split && r.metric === metric);
  return row ? Number(row.value) : null;
}

export function systemsWith(rows: MetricRow[] | undefined, split: string, metric: string): string[] {
  const names = (rows ?? []).filter((r) => r.split === split && r.metric === metric).map((r) => r.system);
  return Array.from(new Set(names));
}

export function percent(value: number | null, digits = 1): string {
  return value === null ? "NOT RUN YET" : `${(100 * value).toFixed(digits)}`;
}

export const SYSTEM_LABELS: Record<string, string> = {
  bm25_default: "BM25 (textbook k1, b)",
  tfidf: "tf-idf cosine (VSM)",
  bm25: "BM25 (tuned on dev)",
  bm25_qr: "BM25 + query reduction",
  bm25_passages: "BM25 on passages",
  bm25_bigram: "BM25 + bigrams",
  "bm25_bigram+citation_pagerank": "+ PageRank (citation graph)",
  "bm25_bigram+similarity_pagerank": "+ PageRank (similarity graph)",
  "bm25_bigram+reference_hub": "+ HITS hub (section graph)",
  "bm25_bigram+local_hits": "+ query-dependent HITS",
  "bm25_bigram+teacher_rerank": "+ cross-encoder re-rank",
  nb: "Naive Bayes",
  rocchio: "Rocchio",
  knn: "kNN",
  neural_attention: "Statute cross-attention",
};

export function label(system: string): string {
  if (SYSTEM_LABELS[system]) return SYSTEM_LABELS[system];
  if (system.startsWith("dense_")) return `Neural (${system.slice(6)})`;
  if (system.startsWith("fusion_weighted_")) return `BM25 + neural (${system.slice(16)})`;
  if (system.startsWith("fusion_rrf_")) return `RRF fusion (${system.slice(11)})`;
  return system;
}
