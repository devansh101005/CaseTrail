"use client";
// Horizontal bar chart in plain SVG: one series, values in %, an optional dashed reference line (published baseline).
import { useState } from "react";

export type Bar = { label: string; value: number };

type Props = { bars: Bar[]; reference?: { label: string; value: number }; caption: string };

export default function BarChart({ bars, reference, caption }: Props) {
  const [hover, setHover] = useState<number | null>(null);
  const rowHeight = 28;
  const labelWidth = 230;
  const width = 720;
  const plotWidth = width - labelWidth - 70;
  const max = Math.max(reference?.value ?? 0, ...bars.map((b) => b.value)) * 1.1 || 1;
  const height = bars.length * rowHeight + 40;
  const x = (v: number) => labelWidth + (v / max) * plotWidth;
  return (
    <figure>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" role="img" aria-label={caption}>
        {bars.map((bar, i) => (
          <g key={bar.label} onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}>
            <rect x={0} y={i * rowHeight} width={width} height={rowHeight} fill={hover === i ? "var(--paper-raised)" : "transparent"} />
            <text x={labelWidth - 8} y={i * rowHeight + 18} textAnchor="end" fontSize="12" fill="var(--ink)">{bar.label}</text>
            <rect x={labelWidth} y={i * rowHeight + 7} width={Math.max(x(bar.value) - labelWidth, 1)} height={14} rx={2} fill="var(--line-red)" />
            {/* Values sit in a fixed column on the right, so they never collide with bars or the reference line */}
            <text x={width - 4} y={i * rowHeight + 18} textAnchor="end" fontSize="11" fontFamily="var(--font-jetbrains)"
              fill="var(--ink)" fontWeight={hover === i ? 700 : 400}>
              {bar.value.toFixed(2)}
            </text>
            <title>{`${bar.label}: ${bar.value.toFixed(2)}`}</title>
          </g>
        ))}
        {reference && (
          <g>
            <line x1={x(reference.value)} x2={x(reference.value)} y1={0} y2={bars.length * rowHeight} stroke="var(--ink)" strokeDasharray="4 3" />
            <text x={x(reference.value)} y={bars.length * rowHeight + 16} textAnchor="middle" fontSize="11" fill="var(--ink)">
              {`${reference.label} ${reference.value}`}
            </text>
          </g>
        )}
        <line x1={labelWidth} x2={labelWidth} y1={0} y2={bars.length * rowHeight} stroke="var(--hairline)" />
      </svg>
      <figcaption className="mt-2 text-xs text-ink-soft">{caption}</figcaption>
    </figure>
  );
}
