// A plain grid table of metrics (systems x metrics) read from the long-format results rows.
import { label, metricValue } from "@/lib/metrics";
import type { MetricRow } from "@/lib/types";

type Column = { metric: string; split: string; title: string; percent?: boolean };

function show(value: number | null, asPercent?: boolean): string {
  if (value === null) return "—";
  return asPercent ? (100 * value).toFixed(2) : String(value);
}

export default function MetricTable({ rows, systems, columns }: { rows: MetricRow[]; systems: string[]; columns: Column[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[12px]">
        <thead>
          <tr className="border-b-2 border-ink">
            <th className="py-2 pr-4 text-left font-semibold">System</th>
            {columns.map((c) => (
              <th key={`${c.split}-${c.metric}`} className="px-2 py-2 text-right font-semibold">{c.title}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {systems.map((system) => (
            <tr key={system} className="border-b border-hairline">
              <td className="py-1.5 pr-4">{label(system)}</td>
              {columns.map((c) => {
                const value = metricValue(rows, system, c.split, c.metric);
                return <td key={`${c.split}-${c.metric}`} className="px-2 py-1.5 text-right">{show(value, c.percent)}</td>;
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
