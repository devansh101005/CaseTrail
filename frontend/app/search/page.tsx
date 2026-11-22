"use client";
import { useEffect, useState } from "react";
import EnginePanel from "@/components/EnginePanel";
import StatRow from "@/components/StatRow";
import StatuteChips from "@/components/StatuteChips";
import TrailTimeline from "@/components/TrailTimeline";
import { getExamples, getHealth, predictStatutes, searchPrecedents } from "@/lib/api";
import type { Example, Health, SearchResponse, StatuteResponse } from "@/lib/types";

export default function SearchPage() {
  const [text, setText] = useState("");
  const [health, setHealth] = useState<Health | null>(null);
  const [examples, setExamples] = useState<Example[]>([]);
  const [system, setSystem] = useState("bm25_bigram");
  const [method, setMethod] = useState("knn");
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [statutes, setStatutes] = useState<StatuteResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    getHealth().then(setHealth).catch((e: Error) => setError(e.message));
    getExamples().then(setExamples).catch(() => setExamples([]));
  }, []);

  async function run() {
    setLoading(true);
    setError("");
    try {
      const [found, predicted] = await Promise.all([searchPrecedents(text, system), predictStatutes(text, method)]);
      setResponse(found);
      setStatutes(predicted);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }

  const top = statutes?.statutes.find((s) => s.above_threshold);
  const cited = response?.results.filter((r) => r.is_relevant_in_benchmark).length;
  return (
    <div className="mx-auto max-w-[1120px] px-4 py-10 md:px-6">
      <p className="mono-label text-line-red-ink">Search</p>
      <h1 className="display mt-2 text-[40px] md:text-[56px]">Plot a trail</h1>
      <label htmlFor="query" className="mono-label mt-8 block">Paste a judgment or describe the facts</label>
      <textarea id="query" value={text} onChange={(e) => setText(e.target.value)} rows={8}
        className="mt-2 w-full border-2 border-ink bg-paper-raised p-3 font-mono text-[13px]" />
      <div className="mt-3 flex flex-wrap gap-2">
        {examples.map((ex) => (
          <button key={ex.id} onClick={() => setText(ex.text)} className="mono-label border border-ink px-2 py-1 text-[10px] hover:bg-paper-raised">
            {ex.title.slice(0, 40)}
          </button>
        ))}
      </div>
      <div className="mt-5 flex flex-wrap items-end gap-4">
        <label className="mono-label">System
          <select value={system} onChange={(e) => setSystem(e.target.value)} className="ml-2 border-2 border-ink bg-paper-raised p-1">
            {(health?.systems ?? ["bm25_bigram"]).map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        <label className="mono-label">Statutes
          <select value={method} onChange={(e) => setMethod(e.target.value)} className="ml-2 border-2 border-ink bg-paper-raised p-1">
            {(health?.methods ?? ["knn"]).map((m) => <option key={m} value={m}>{m}</option>)}
          </select>
        </label>
        <button onClick={run} disabled={loading || !text.trim()} className="mono-label bg-ink px-5 py-3 text-on-ink disabled:opacity-40">
          {loading ? "Plotting the trail…" : "Search"}
        </button>
      </div>
      {error && <p className="mono-label mt-4 text-line-red-ink" role="alert">{error}</p>}
      {response && statutes && (
        <div className="mt-10">
          <StatRow stats={[
            { label: "Results", value: String(response.results.length) },
            { label: "MS", value: response.timings_ms.total.toFixed(0) },
            { label: "Top statute", value: top ? top.id.replace("Section ", "IPC ") : "none" },
            { label: "Cited (benchmark)", value: cited === undefined || response.known_query_id === null ? "—" : `${cited}/${response.results.length}` },
          ]} />
          <div className="mt-8 grid gap-8 md:grid-cols-[2fr_1fr]">
            <TrailTimeline results={response.results} />
            <div className="flex flex-col gap-6">
              <div>
                <p className="mono-label mb-3 text-line-amber-ink">Statutes</p>
                <StatuteChips statutes={statutes.statutes} method={statutes.method} />
              </div>
              <EnginePanel response={response} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
