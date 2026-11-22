// Four labelled values separated by hairlines (Waypoint "stat row").
export type Stat = { label: string; value: string };

export default function StatRow({ stats }: { stats: Stat[] }) {
  return (
    <div className="grid grid-cols-2 border-y-2 border-ink md:grid-cols-4">
      {stats.map((stat) => (
        <div key={stat.label} className="border-r border-hairline p-4 last:border-r-0">
          <p className="mono-label text-ink-soft">{stat.label}</p>
          <p className="mt-1 font-mono text-[26px] font-bold">{stat.value}</p>
        </div>
      ))}
    </div>
  );
}
