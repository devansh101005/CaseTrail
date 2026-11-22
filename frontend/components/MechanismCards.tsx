const CARDS = [
  { n: "01", title: "Keyword", colour: "bg-line-red", text: "BM25 over our own inverted index, with word bigrams, legal section tokens and dev-tuned k1, b." },
  { n: "02", title: "Meaning", colour: "bg-line-blue", text: "A small bi-encoder compares judgments chunk by chunk (MaxSim), exported to INT8 ONNX for the CPU." },
  { n: "03", title: "Authority", colour: "bg-line-green", text: "PageRank and HITS on judgment graphs: citation links from training labels, BM25 similarity, sections." },
  { n: "04", title: "Statutes", colour: "bg-line-amber", text: "Naive Bayes, Rocchio and kNN over 100 IPC sections, with per-statute thresholds tuned on dev." },
];

export default function MechanismCards() {
  return (
    <section className="bg-paper">
      <div className="mx-auto max-w-[1120px] px-4 py-16 md:px-6">
        <p className="mono-label mb-3 text-line-red-ink">How the trail is built</p>
        <h2 className="display mb-10 text-[34px] md:text-[48px]">Four signals, one trail</h2>
        <div className="grid gap-0 border-t-2 border-ink md:grid-cols-4">
          {CARDS.map((card) => (
            <article key={card.n} className="border-b border-hairline p-5 md:border-r md:last:border-r-0">
              <span className={`mono-label inline-block px-2 py-1 text-white ${card.colour}`}>{card.n}</span>
              <h3 className="display mt-4 text-[20px]">{card.title}</h3>
              <p className="mt-3 font-mono text-[13px] leading-relaxed">{card.text}</p>
            </article>
          ))}
        </div>
      </div>
    </section>
  );
}
