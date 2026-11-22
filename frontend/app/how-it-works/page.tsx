// The pipeline and one card per component, each naming its source file (no numbers on this page).
const STEPS = [
  { title: "Text pipeline", file: "src/casetrail/text/", colour: "border-ink",
    text: "Strips leftover citations (they leak the answers), drops mask tags, lower-cases, protects legal references as tokens (sec_498a, art_21), removes English and legal boilerplate stop words, light stemming." },
  { title: "Inverted index", file: "src/casetrail/index/inverted_index.py", colour: "border-line-red",
    text: "Dictionary + postings with term frequencies in flat numpy arrays; one index per candidate pool, with word bigrams as extra terms." },
  { title: "BM25 / tf-idf", file: "src/casetrail/retrieval/bm25.py", colour: "border-line-red",
    text: "Term-at-a-time score accumulators and a heap for the top K; query reduction keeps the strongest query terms by tf × idf." },
  { title: "Link analysis", file: "src/casetrail/graph/", colour: "border-line-green",
    text: "PageRank (power iteration, damping, dangling nodes) on a citation graph built from training labels and on a BM25 similarity graph; HITS on judgment–section graphs and per query." },
  { title: "Neural retriever", file: "src/casetrail/neural/", colour: "border-line-blue",
    text: "Chunked bi-encoder with MaxSim scoring, trained on Kaggle with BM25 hard negatives and distilled from a cross-encoder, exported to INT8 ONNX." },
  { title: "Fusion", file: "src/casetrail/fusion/combine.py", colour: "border-line-blue",
    text: "Reciprocal rank fusion and a weighted sum of normalised scores; the weight is chosen on dev." },
  { title: "Statute classifiers", file: "src/casetrail/classify/", colour: "border-line-amber",
    text: "Multinomial Naive Bayes, Rocchio centroids and kNN over tf-idf, one-vs-rest over 100 IPC sections with per-statute thresholds tuned on dev." },
  { title: "Evaluation", file: "src/casetrail/eval/", colour: "border-ink",
    text: "micro-F1@K (IL-TUR protocol), P@K, R@K, MAP, nDCG, macro/micro-F1 and paired randomisation tests; every number is written to results/." },
];

function Pipeline() {
  const boxes = ["Judgment", "Tokens", "Index", "BM25", "Fusion", "Precedents"];
  return (
    <svg viewBox="0 0 720 150" className="w-full" role="img" aria-label="Pipeline from judgment to precedents and statutes">
      {boxes.map((box, i) => (
        <g key={box}>
          <rect x={10 + i * 118} y={20} width={100} height={40} fill="var(--paper-raised)" stroke="var(--ink)" strokeWidth={2} />
          <text x={60 + i * 118} y={45} textAnchor="middle" fontSize={12} fontFamily="var(--font-jetbrains)">{box.toUpperCase()}</text>
          {i < boxes.length - 1 && <line x1={110 + i * 118} x2={128 + i * 118} y1={40} y2={40} stroke="var(--ink)" strokeWidth={2} />}
        </g>
      ))}
      <path d="M 60 60 V 110 H 482 V 60" fill="none" stroke="var(--line-blue)" strokeWidth={3} />
      <text x={260} y={128} textAnchor="middle" fontSize={11} fill="var(--line-blue)">neural encoder (chunks → vectors → MaxSim)</text>
      <path d="M 400 20 V 6 H 482 V 20" fill="none" stroke="var(--line-green)" strokeWidth={3} />
      <text x={440} y={4} textAnchor="middle" fontSize={10} fill="var(--line-green)">graph authority</text>
      <path d="M 60 60 V 140 H 690" fill="none" stroke="var(--line-amber)" strokeWidth={3} strokeDasharray="6 4" />
      <text x={600} y={136} textAnchor="middle" fontSize={11} fill="var(--line-amber-ink)">statute classifiers → IPC sections</text>
    </svg>
  );
}

export default function HowItWorks() {
  return (
    <div className="mx-auto max-w-[1120px] px-4 py-10 md:px-6">
      <p className="mono-label text-line-red-ink">How it works</p>
      <h1 className="display mt-2 text-[40px] md:text-[56px]">How the trail is built</h1>
      <div className="mt-8"><Pipeline /></div>
      <div className="mt-10 grid gap-6 md:grid-cols-2">
        {STEPS.map((step) => (
          <article key={step.title} className={`border-l-4 ${step.colour} bg-paper-raised p-5`}>
            <h2 className="display text-[18px]">{step.title}</h2>
            <p className="mono-label mt-1 text-[10px] text-ink-soft">{step.file}</p>
            <p className="mt-3 text-sm leading-relaxed">{step.text}</p>
          </article>
        ))}
      </div>
    </div>
  );
}
