import Link from "next/link";
import LineDiagram from "./LineDiagram";

export default function Hero() {
  return (
    <section className="bg-paper">
      <div className="mx-auto grid max-w-[1120px] gap-10 px-4 py-16 md:grid-cols-[3fr_2fr] md:px-6 md:py-24">
        <div>
          <p className="mono-label mb-4 text-line-red-ink">Precedent &amp; statute search</p>
          <h1 className="display text-[44px] md:text-[72px]">
            Every judgment is a station. <span className="text-line-red">Precedent is the trail.</span>
          </h1>
          <p className="mt-6 max-w-[60ch] text-base leading-relaxed">
            Paste an Indian Supreme Court judgment or describe the facts. CaseTrail returns earlier judgments that
            the case is likely to cite, the IPC sections that probably apply, and how much each signal (keywords,
            meaning, graph authority) contributed. A research aid, not legal advice.
          </p>
          <div className="mt-8 flex flex-wrap gap-3">
            <Link href="/search" className="mono-label bg-ink px-5 py-3 text-on-ink hover:bg-ink-soft">
              Search a judgment
            </Link>
            <Link href="/evidence" className="mono-label border-2 border-ink px-5 py-3 hover:bg-paper-raised">
              See the evidence
            </Link>
          </div>
          <Link href="/how-it-works" className="mono-label mt-5 inline-block text-line-red-ink underline">
            How the trail is built →
          </Link>
        </div>
        <div className="flex items-center justify-center">
          <LineDiagram />
        </div>
      </div>
    </section>
  );
}
