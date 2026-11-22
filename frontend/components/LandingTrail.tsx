"use client";
// "A trail, plotted": the first real IL-TUR test judgment from /examples and its top precedents, live from /search.
import Link from "next/link";
import { useEffect, useState } from "react";
import { getExamples, searchPrecedents } from "@/lib/api";
import type { Example, SearchResponse } from "@/lib/types";

export default function LandingTrail() {
  const [example, setExample] = useState<Example | null>(null);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    getExamples(1)
      .then((found) => {
        if (found.length === 0) throw new Error("no examples returned by the API");
        setExample(found[0]);
        return searchPrecedents(found[0].text, "bm25_bigram", 5);
      })
      .then(setResponse)
      .catch((e: Error) => setError(e.message));
  }, []);

  const cited = response?.results.filter((r) => r.is_relevant_in_benchmark).length ?? 0;
  return (
    <section className="dotted-grid bg-ink text-on-ink">
      <div className="mx-auto max-w-[1120px] px-4 py-16 md:px-6">
        <p className="mono-label mb-3 text-red-on-ink">A trail, plotted</p>
        <h2 className="display mb-6 text-[34px] md:text-[48px]">One real judgment, five stops</h2>
        {error && <p className="mono-label text-red-on-ink">{error}</p>}
        {!response && !error && <p className="mono-label">Plotting the trail…</p>}
        {example && response && (
          <>
            <p className="max-w-[70ch] text-sm text-on-ink-muted">
              Query: IL-TUR test judgment {example.id} ({example.title}). The real judgment cites{" "}
              {example.num_relevant} earlier cases; {cited} of them are among the five stops below.
            </p>
            <ol className="mt-8 border-l-2 border-dashed border-on-ink-muted" style={{ marginLeft: 8 }}>
              {response.results.map((r) => (
                <li key={r.doc_id} className="relative pb-6 pl-8">
                  <span className={`absolute -left-[9px] top-1 h-4 w-4 border-[3px] border-on-ink ${r.rank === 1 ? "bg-on-ink" : "bg-ink"}`} aria-hidden="true" />
                  <p className="display text-[16px] normal-case">{r.title}</p>
                  <p className="mono-label mt-1 text-[10px] text-on-ink-muted">
                    Stop {String(r.rank).padStart(2, "0")} · BM25 {r.breakdown.bm25.toFixed(1)}
                    {r.is_relevant_in_benchmark ? " · cited in the real judgment" : ""}
                  </p>
                </li>
              ))}
            </ol>
            <Link href="/search" className="mono-label inline-block bg-line-red px-5 py-3 text-white">Plot your own →</Link>
          </>
        )}
      </div>
    </section>
  );
}
