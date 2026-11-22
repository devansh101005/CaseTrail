"use client";
// "Measured, not claimed": headline numbers from results/ next to the published baselines they compare with.
import { useEffect, useState } from "react";
import { getResults } from "@/lib/api";
import { metricValue, percent } from "@/lib/metrics";
import type { ResultsPayload } from "@/lib/types";

type Tile = { label: string; value: string; compare: string; colour: string };

function tiles(results: ResultsPayload): Tile[] {
  const baselines = results.published_baselines?.baselines ?? [];
  const sota = baselines.find((b) => b.task === "pcr" && b.system.includes("SOTA"));
  const lesicin = baselines.find((b) => b.task === "lsi");
  const lsiBest = ["knn", "nb", "neural_attention"]
    .map((m) => ({ m, v: metricValue(results.lsi_metrics, m, "dev", "macro_f1") }))
    .filter((x) => x.v !== null)
    .sort((a, b) => (b.v ?? 0) - (a.v ?? 0))[0];
  const efficiency = results.efficiency ?? [];
  const int8 = efficiency.find((r) => r.system.endsWith("_onnx_int8") && r.metric === "size_mb");
  const latency = efficiency.find((r) => r.system.endsWith("_onnx_int8") && r.metric === "query_encode_median_ms");
  return [
    { label: "Micro-F1@K (ours, BM25 + bigrams)", value: percent(metricValue(results.pcr_metrics, "bm25_bigram", "test", "micro_f1@k")),
      compare: sota ? `published SOTA: ${sota.value_percent}` : "baseline: not loaded", colour: "text-red-on-ink" },
    { label: `Macro-F1 (statutes, ${lsiBest?.m ?? "—"})`,
      value: lsiBest ? percent(metricValue(results.lsi_metrics, lsiBest.m, "test", "macro_f1")) : "NOT RUN YET",
      compare: lesicin ? `LeSICiN: ${lesicin.value_percent}` : "baseline: not loaded", colour: "text-line-amber" },
    { label: "Student model (INT8)", value: int8 ? `${Number(int8.value).toFixed(0)} MB` : "NOT RUN YET",
      compare: "ONNX Runtime, dynamic quantisation", colour: "text-on-ink" },
    { label: "Neural query encode (CPU)", value: latency ? `${Number(latency.value).toFixed(0)} MS` : "NOT RUN YET",
      compare: "median, one judgment", colour: "text-on-ink" },
  ];
}

export default function EvidenceTiles() {
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getResults().then(setResults).catch((e: Error) => setError(e.message));
  }, []);
  return (
    <section className="dotted-grid bg-ink text-on-ink">
      <div className="mx-auto max-w-[1120px] px-4 py-16 md:px-6">
        <p className="mono-label mb-3 text-red-on-ink">Measured, not claimed</p>
        <h2 className="display mb-10 text-[34px] md:text-[48px]">The evidence</h2>
        {error && <p className="mono-label text-red-on-ink">{error}</p>}
        {!results && !error && <p className="mono-label">Plotting the trail…</p>}
        {results && (
          <div className="grid gap-px bg-hairline-ink md:grid-cols-4">
            {tiles(results).map((tile) => (
              <div key={tile.label} className="bg-ink p-5">
                <p className={`font-mono text-[44px] font-bold ${tile.colour}`}>{tile.value}</p>
                <p className="mono-label mt-2 text-on-ink-muted">{tile.compare}</p>
                <p className="mono-label mt-4">{tile.label}</p>
              </div>
            ))}
          </div>
        )}
        <p className="mt-6 max-w-[70ch] text-sm text-on-ink-muted">
          F1 scores are percentages on the IL-TUR test split, with every setting chosen on dev; size and time are measured on this laptop. Published numbers come
          from the IL-TUR paper (Joshi et al., 2024); see the evidence page for the full tables and caveats.
        </p>
      </div>
    </section>
  );
}
