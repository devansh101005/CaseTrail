"use client";
// Real numbers from /results, scrolling slowly (stopped under prefers-reduced-motion).
import { useEffect, useState } from "react";
import { getResults } from "@/lib/api";
import { metricValue, percent } from "@/lib/metrics";
import type { ResultsPayload } from "@/lib/types";

function items(results: ResultsPayload | null): string[] {
  if (!results) return ["LOADING RESULTS…"];
  const pcrTest = metricValue(results.pcr_metrics, "bm25_bigram", "test", "micro_f1@k");
  const queries = results.pcr_metrics?.find((r) => r.split === "test")?.n;
  const lsiKnn = metricValue(results.lsi_metrics, "knn", "test", "macro_f1");
  const stats = results.data_stats as { lsi?: { statutes?: number } } | undefined;
  return [
    `${queries ?? "NOT RUN YET"} TEST QUERIES`,
    `BM25 + BIGRAMS MICRO-F1@K ${percent(pcrTest)}`,
    `${stats?.lsi?.statutes ?? "NOT RUN YET"} IPC SECTIONS`,
    `kNN STATUTE MACRO-F1 ${percent(lsiKnn)}`,
    "TUNED ON DEV · SCORED ONCE ON TEST",
  ];
}

export default function Ticker() {
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [failed, setFailed] = useState(false);
  useEffect(() => {
    getResults().then(setResults).catch(() => setFailed(true));
  }, []);
  const line = failed ? ["API OFFLINE: START uvicorn api.main:app"] : items(results);
  const text = line.map((item) => `◆ ${item} `).join("");
  return (
    <div className="overflow-hidden border-y-2 border-ink bg-paper-raised py-3" aria-label={line.join(", ")}>
      <div className="ticker-track flex w-max whitespace-nowrap">
        <span className="mono-label px-4 text-[12px]">{text}</span>
        <span className="mono-label px-4 text-[12px]" aria-hidden="true">{text}</span>
      </div>
    </div>
  );
}
