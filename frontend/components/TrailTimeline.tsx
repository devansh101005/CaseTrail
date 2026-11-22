"use client";
// Each retrieved judgment is a station on a dashed line; "WHY THIS CASE?" opens its signal breakdown.
import { useState } from "react";
import type { ResultItem } from "@/lib/types";

function fmt(value: number | null, digits: number): string {
  return value === null ? "—" : value.toFixed(digits);
}

function Station({ item }: { item: ResultItem }) {
  const [open, setOpen] = useState(false);
  const b = item.breakdown;
  return (
    <li className="relative pb-8 pl-10">
      <span className={`absolute left-[7px] top-1 h-4 w-4 border-[3px] border-ink ${item.rank === 1 ? "bg-ink" : "bg-paper"}`} aria-hidden="true" />
      <div className="flex flex-wrap items-baseline gap-2">
        <h3 className="display text-[18px] normal-case">{item.title}</h3>
        {item.is_relevant_in_benchmark && (
          <span className="mono-label border border-line-green px-2 py-0.5 text-line-green">Cited in the real judgment</span>
        )}
      </div>
      <p className="mono-label mt-1 text-[10px] text-ink-soft">
        Rank {String(item.rank).padStart(2, "0")} · ID {item.doc_id} · BM25 {fmt(b.bm25, 1)}
        {b.neural !== null && ` · Neural ${fmt(b.neural, 3)}`}
        {b.pagerank_citation !== null && ` · PageRank ${b.pagerank_citation.toExponential(2)}`}
      </p>
      <p className="mt-2 max-w-[75ch] text-sm leading-relaxed">{item.snippet}…</p>
      <button onClick={() => setOpen(!open)} className="mono-label mt-3 border-2 border-ink px-3 py-1 hover:bg-paper-raised" aria-expanded={open}>
        {open ? "Hide" : "Why this case?"}
      </button>
      {open && (
        <div className="mt-3 border-l-4 border-line-red bg-paper-raised p-4">
          <dl className="grid grid-cols-2 gap-x-6 gap-y-1 font-mono text-[12px] md:grid-cols-3">
            <dt className="text-line-red-ink">BM25 (keyword)</dt><dd>{fmt(b.bm25, 2)}</dd><dd className="hidden md:block" />
            <dt className="text-line-blue">Neural (meaning)</dt><dd>{fmt(b.neural, 4)}</dd><dd className="hidden md:block" />
            <dt>Fused score</dt><dd>{fmt(b.fusion, 4)}</dd><dd className="hidden md:block" />
            <dt className="text-line-green">PageRank (cited in training)</dt><dd>{b.pagerank_citation === null ? "not in graph" : b.pagerank_citation.toExponential(3)}</dd><dd className="hidden md:block" />
            <dt className="text-line-green">PageRank (similarity graph)</dt><dd>{b.pagerank_similarity === null ? "—" : b.pagerank_similarity.toExponential(3)}</dd>
          </dl>
          {item.matched_passages.length > 0 && <p className="mono-label mt-4">Best-matching passages</p>}
          {item.matched_passages.map((passage, i) => (
            <blockquote key={i} className="mt-2 border-l-2 border-hairline pl-3 text-sm">{passage}…</blockquote>
          ))}
        </div>
      )}
    </li>
  );
}

export default function TrailTimeline({ results }: { results: ResultItem[] }) {
  if (results.length === 0) return <p className="mono-label">No precedents found.</p>;
  return (
    <ol className="relative border-l-2 border-dashed border-ink pl-0" style={{ marginLeft: 14 }}>
      {results.map((item) => (
        <Station key={item.doc_id} item={item} />
      ))}
    </ol>
  );
}
