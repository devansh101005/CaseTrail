// Three signal "lines" (keyword, meaning, authority) meeting at interchange judgments. Decorative, no data.
export default function LineDiagram() {
  return (
    <svg viewBox="0 0 320 260" role="img" aria-label="Three coloured lines joining at shared judgments" className="w-full max-w-[360px]">
      <path d="M10 40 H120 L200 120 H310" fill="none" stroke="var(--line-red)" strokeWidth="6" />
      <path d="M10 120 H310" fill="none" stroke="var(--line-blue)" strokeWidth="6" />
      <path d="M10 210 H110 L200 120 L250 70 H310" fill="none" stroke="var(--line-green)" strokeWidth="6" />
      {[60, 150, 260].map((x) => (
        <circle key={x} cx={x} cy="120" r="7" fill="var(--paper)" stroke="var(--ink)" strokeWidth="3" />
      ))}
      <rect x="190" y="110" width="20" height="20" transform="rotate(45 200 120)" fill="var(--ink)" />
      <text x="10" y="30" className="mono-label" fill="var(--line-red-ink)">Keyword</text>
      <text x="10" y="110" className="mono-label" fill="var(--line-blue)">Meaning</text>
      <text x="10" y="200" className="mono-label" fill="var(--line-green)">Authority</text>
    </svg>
  );
}
