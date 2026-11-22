// "What the engine did": the reduced BM25 query (terms with df / idf) and per-stage timings.
import type { SearchResponse } from "@/lib/types";

export default function EnginePanel({ response }: { response: SearchResponse }) {
  return (
    <aside className="bg-ink p-5 text-on-ink">
      <p className="mono-label text-red-on-ink">What the engine did</p>
      <p className="mt-3 text-sm text-on-ink-muted">
        {response.num_query_terms} distinct index terms in the query; BM25 kept the strongest by tf × idf.
        {response.known_query_id && ` Recognised as IL-TUR test query ${response.known_query_id}.`}
      </p>
      <table className="mt-4 w-full font-mono text-[12px]">
        <thead>
          <tr className="text-on-ink-muted">
            <th className="text-left font-normal">Term</th>
            <th className="text-right font-normal">df</th>
            <th className="text-right font-normal">idf</th>
          </tr>
        </thead>
        <tbody>
          {response.query_terms.map((t) => (
            <tr key={t.term} className="border-t border-hairline-ink">
              <td className="py-1">{t.term}</td>
              <td className="text-right">{t.df}</td>
              <td className="text-right">{t.idf.toFixed(2)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="mono-label mt-5 text-red-on-ink">Timings (ms)</p>
      <ul className="mt-2 font-mono text-[12px]">
        {Object.entries(response.timings_ms).map(([stage, ms]) => (
          <li key={stage} className="flex justify-between border-t border-hairline-ink py-1">
            <span>{stage}</span>
            <span>{ms.toFixed(0)}</span>
          </li>
        ))}
      </ul>
    </aside>
  );
}
