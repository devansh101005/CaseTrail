"use client";
import { useEffect, useState } from "react";
import BarChart from "@/components/BarChart";
import MetricTable from "@/components/MetricTable";
import { getResults } from "@/lib/api";
import { label, metricValue, systemsWith } from "@/lib/metrics";
import type { ResultsPayload } from "@/lib/types";

const PCR_COLUMNS = [
  { metric: "micro_f1@k", split: "dev", title: "dev µF1@K", percent: true },
  { metric: "k", split: "test", title: "K" },
  { metric: "micro_f1@k", split: "test", title: "test µF1@K", percent: true },
  { metric: "micro_precision@k", split: "test", title: "P@K", percent: true },
  { metric: "micro_recall@k", split: "test", title: "R@K", percent: true },
  { metric: "map@100", split: "test", title: "MAP", percent: true },
  { metric: "ndcg@10", split: "test", title: "nDCG@10", percent: true },
];
const LSI_COLUMNS = [
  { metric: "macro_f1", split: "dev", title: "dev macro-F1", percent: true },
  { metric: "macro_f1", split: "test", title: "test macro-F1", percent: true },
  { metric: "macro_precision", split: "test", title: "macro-P", percent: true },
  { metric: "macro_recall", split: "test", title: "macro-R", percent: true },
  { metric: "micro_f1", split: "test", title: "micro-F1", percent: true },
];

function Section({ title, how, takeaway, children }: { title: string; how: string; takeaway: string; children: React.ReactNode }) {
  return (
    <section className="border-t-2 border-ink py-10">
      <h2 className="display text-[28px]">{title}</h2>
      <p className="mt-2 text-sm"><span className="mono-label">How to read: </span>{how}</p>
      <div className="mt-6">{children}</div>
      <p className="mt-4 text-sm"><span className="mono-label text-line-red-ink">Takeaway: </span>{takeaway}</p>
    </section>
  );
}

export default function EvidencePage() {
  const [results, setResults] = useState<ResultsPayload | null>(null);
  const [error, setError] = useState("");
  useEffect(() => {
    getResults().then(setResults).catch((e: Error) => setError(e.message));
  }, []);
  if (error) return <p className="mono-label mx-auto max-w-[1120px] p-6 text-line-red-ink">{error}</p>;
  if (!results) return <p className="mono-label mx-auto max-w-[1120px] p-6">Plotting the trail…</p>;

  const pcr = results.pcr_metrics ?? [];
  const lsi = results.lsi_metrics ?? [];
  const pcrSystems = systemsWith(pcr, "test", "micro_f1@k").filter((s) => !s.includes("citations_kept"));
  const bestDev = [...pcrSystems].sort((a, b) => (metricValue(pcr, b, "dev", "micro_f1@k") ?? 0) - (metricValue(pcr, a, "dev", "micro_f1@k") ?? 0))[0];
  const sota = results.published_baselines?.baselines.find((b) => b.task === "pcr" && b.system.includes("SOTA"));
  const lesicin = results.published_baselines?.baselines.find((b) => b.task === "lsi");
  const lsiSystems = systemsWith(lsi, "test", "macro_f1");
  const bestLsi = [...lsiSystems].sort((a, b) => (metricValue(lsi, b, "dev", "macro_f1") ?? 0) - (metricValue(lsi, a, "dev", "macro_f1") ?? 0))[0];
  const pcrBars = pcrSystems.map((s) => ({ label: label(s), value: 100 * (metricValue(pcr, s, "test", "micro_f1@k") ?? 0) }))
    .sort((a, b) => b.value - a.value);
  const lsiBars = lsiSystems.map((s) => ({ label: label(s), value: 100 * (metricValue(lsi, s, "test", "macro_f1") ?? 0) }))
    .sort((a, b) => b.value - a.value);

  return (
    <div className="mx-auto max-w-[1120px] px-4 py-10 md:px-6">
      <p className="mono-label text-line-red-ink">Evidence</p>
      <h1 className="display mt-2 text-[40px] md:text-[56px]">Measured, not claimed</h1>
      <p className="mt-4 max-w-[70ch] text-sm">Every number below is read from <code>results/</code>, written by the scripts in the repository. All settings were chosen on dev; test was scored once per system.</p>
      <Section title="Prior case retrieval"
        how="micro-F1@K pools the top-K candidates of every test query (IL-TUR / U-CREAT protocol); K was chosen on dev."
        takeaway={bestDev ? `Best system on dev: ${label(bestDev)}, test µF1@K ${(100 * (metricValue(pcr, bestDev, "test", "micro_f1@k") ?? 0)).toFixed(2)}${sota ? ` vs the published state of the art ${sota.value_percent}` : ""}. Differences under about one point between the BM25 + bigram variants are not significant (see below); published numbers may use different preprocessing.` : "NOT RUN YET"}>
        {pcrBars.length > 0 && <BarChart bars={pcrBars} reference={sota ? { label: "published SOTA", value: sota.value_percent } : undefined} caption="Test micro-F1@K (%) per system" />}
        <div className="mt-6"><MetricTable rows={pcr} systems={pcrSystems} columns={PCR_COLUMNS} /></div>
      </Section>
      <Section title="Statute identification"
        how="macro-F1 averages the F1 of each of the 100 IPC sections, so rare sections count as much as common ones."
        takeaway={bestLsi ? `Best on dev: ${label(bestLsi)}, test macro-F1 ${(100 * (metricValue(lsi, bestLsi, "test", "macro_f1") ?? 0)).toFixed(2)}${lesicin ? ` vs LeSICiN ${lesicin.value_percent}` : ""}.` : "NOT RUN YET"}>
        {lsiBars.length > 0 && <BarChart bars={lsiBars} reference={lesicin ? { label: "LeSICiN", value: lesicin.value_percent } : undefined} caption="Test macro-F1 (%) per method" />}
        <div className="mt-6"><MetricTable rows={lsi} systems={lsiSystems} columns={LSI_COLUMNS} /></div>
      </Section>
      <Section title="Significance"
        how="Paired randomisation tests on test: the p-value is the chance of a difference this large if the two systems were equally good."
        takeaway={(results.significance ?? []).length ? "p < 0.05 marks a difference unlikely to be chance." : "NOT RUN YET"}>
        <div className="overflow-x-auto">
          <table className="w-full font-mono text-[12px]">
            <thead>
              <tr className="border-b-2 border-ink text-left">
                <th className="py-1 font-semibold">Comparison (A vs B)</th>
                <th className="font-semibold">Metric</th>
                <th className="text-right font-semibold">A − B (points)</th>
                <th className="text-right font-semibold">p-value</th>
              </tr>
            </thead>
            <tbody>
              {(results.significance ?? []).map((row, i) => (
                <tr key={i} className="border-b border-hairline">
                  <td className="py-1">{label(row.system_a)} vs {label(row.system_b)}</td>
                  <td>{row.metric}</td>
                  <td className="text-right">{(100 * Number(row.difference)).toFixed(2)}</td>
                  <td className="text-right">{row.p_value}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>
    </div>
  );
}
