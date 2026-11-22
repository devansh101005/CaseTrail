import Link from "next/link";

const NAV = [
  { href: "/search", label: "Search" },
  { href: "/evidence", label: "Evidence" },
  { href: "/how-it-works", label: "How it works" },
  { href: "/team", label: "Team" },
];

export default function TopBar() {
  return (
    <header className="bg-ink text-on-ink">
      <div className="mx-auto flex max-w-[1120px] flex-wrap items-center gap-4 px-4 py-3 md:px-6">
        <Link href="/" className="mono-label text-[13px] tracking-[0.28em] text-on-ink">
          CASETRAIL
        </Link>
        <span className="mono-label bg-line-red px-2 py-1 text-[10px] text-white">IL-TUR · Supreme Court</span>
        <nav className="ml-auto flex flex-wrap gap-4">
          {NAV.map((item) => (
            <Link key={item.href} href={item.href} className="mono-label text-on-ink-muted hover:text-on-ink">
              {item.label}
            </Link>
          ))}
        </nav>
      </div>
    </header>
  );
}
