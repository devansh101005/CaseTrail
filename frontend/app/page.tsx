import Link from "next/link";
import EvidenceTiles from "@/components/EvidenceTiles";
import Hero from "@/components/Hero";
import MechanismCards from "@/components/MechanismCards";
import Ticker from "@/components/Ticker";

export default function Home() {
  return (
    <>
      <Hero />
      <Ticker />
      <MechanismCards />
      <EvidenceTiles />
      <section className="bg-ink text-on-ink">
        <div className="mx-auto flex max-w-[1120px] flex-wrap items-center justify-between gap-6 border-t-2 border-on-ink px-4 py-14 md:px-6">
          <h2 className="display text-[36px] md:text-[56px]">Mind the precedent.</h2>
          <Link href="/search" className="mono-label bg-line-red px-6 py-4 text-white">Search a judgment →</Link>
        </div>
      </section>
    </>
  );
}
