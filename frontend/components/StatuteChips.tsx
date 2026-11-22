// IPC sections with their classifier score and the statute's own dev-tuned decision threshold.
import type { StatuteRow } from "@/lib/types";

export default function StatuteChips({ statutes, method }: { statutes: StatuteRow[]; method: string }) {
  return (
    <div>
      <ul className="flex flex-col gap-2">
        {statutes.map((s) => (
          <li key={s.id} className={`border-2 px-3 py-2 ${s.above_threshold ? "border-line-amber" : "border-hairline"}`}>
            <div className="flex items-baseline justify-between gap-3">
              <span className="mono-label text-line-amber-ink">IPC {s.id.replace("Section ", "")}</span>
              <span className="font-mono text-[12px]">
                {s.probability !== null ? `p = ${s.probability.toFixed(2)}` : `score ${s.score.toFixed(3)}`}
              </span>
            </div>
            <p className="mt-1 text-[13px] leading-snug">{s.title}</p>
            <p className="mono-label mt-1 text-[10px] text-ink-soft">
              {s.above_threshold ? "Predicted" : "Below threshold"} · threshold {s.threshold.toFixed(3)}
            </p>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-xs text-ink-soft">
        Method: {method}. Each statute has its own threshold, tuned on the IL-TUR dev split to maximise its F1.
      </p>
    </div>
  );
}
