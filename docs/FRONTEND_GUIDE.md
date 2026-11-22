# Frontend Guide — CaseTrail

The demo frontend copies the **design language of our Waypoint project** (https://waypoint-six-teal.vercel.app/, my own hackathon app), adapted to legal search.
Waypoint's idea was *"every course is a station, learning is the line between"* (a metro map). CaseTrail's version: **every judgment is a station, and precedent is the trail between them.** The coloured "lines" are our ranking signals.

**The frontend is a display layer only.** It calls the FastAPI backend and shows what it returns. Every number on screen comes from the API or from `results/` — never typed into the code. If a result doesn't exist yet, show `NOT RUN YET`.

---

## 1. Stack
- **Next.js (App Router) + TypeScript + Tailwind CSS**, fonts via `next/font/google` (self-hosted at build time, so the demo works **offline**).
- Small, typed components in `frontend/components/`. API calls in `frontend/lib/api.ts` only.
- Charts: plain hand-written SVG components. No UI kits or chart libraries unless we explicitly agree (see our coding rules).
- Backend URL from `NEXT_PUBLIC_API_URL` (default `http://127.0.0.1:8000`).

## 2. Design tokens (measured from Waypoint)
```css
:root {
  /* surfaces */
  --paper:        #F4F3EF;   /* page background (light sections) */
  --paper-raised: #FBFAF7;   /* inputs, raised panels on paper */
  --ink:          #16161A;   /* text on paper; background of dark sections + top bar */
  --ink-soft:     #2A2A30;   /* panels inside dark sections */
  --on-ink:       #F4F3EF;   /* text on dark sections */
  --on-ink-muted: rgba(244, 243, 239, 0.72);
  --hairline:     #DCDAD3;   /* borders/rules on paper */
  --hairline-ink: rgba(244, 243, 239, 0.22); /* borders on dark */

  /* the "lines" (signal colours) */
  --line-red:    #D82A24;   /* BM25 / keyword line, primary accent, CTA */
  --line-red-ink:#AE1B16;   /* small red labels/kickers on paper */
  --line-blue:   #1B45C4;   /* neural line */
  --line-green:  #00734A;   /* graph line (PageRank / HITS) */
  --line-amber:  #E39A0C;   /* statutes line; numbers on dark */
  --line-amber-ink: #8A5B00;
  --red-on-ink:  #FF6B63;   /* red used on dark backgrounds */

  --radius: 0px;            /* everything is square */
}
```
Optional dark mode: swap paper/ink; keep the line colours.

## 3. Typography
| Role | Font | Style |
|---|---|---|
| Display headings (h1/h2/h3) | **Archivo**, weight 800 | UPPERCASE, tight tracking (h1 ≈ 72px, letter-spacing ≈ -0.025em, line-height ~0.95; h2 ≈ 48px; h3 ≈ 18–20px) |
| Body text | **Inter**, 400 | 16px, line-height 1.6, sentence case, max ~65 characters per line |
| Labels, kickers, nav, metadata, numbers | **JetBrains Mono**, 600–700 | UPPERCASE, 11–12px, very wide tracking (≈ 0.18–0.28em) |
| Big metric numbers | JetBrains Mono 500–700 | 40–56px, coloured by line |

Highlight part of a headline in `--line-red` (Waypoint style: second half of the hero in red).

## 4. Layout
- Max content width ~1120px, generous side padding (≥ 24px; 16px on mobile).
- Sections alternate **paper** and **ink** backgrounds. Dark sections may use a subtle **dotted grid** background (`radial-gradient` dots at ~16px spacing, low opacity).
- Thin hairline rules between columns; thick (2px) ink rules above section footers.
- Must work at 375px wide (stack columns, no horizontal scroll).

## 5. Components (with Waypoint equivalents)
| Component | Look | CaseTrail use |
|---|---|---|
| **TopBar** | ink background, wordmark `CASETRAIL` in mono caps with wide tracking, a red pill-shaped-but-square badge (`LINE 01 · DATA` in Waypoint), mono nav links right-aligned | badge: `IL-TUR · SUPREME COURT`; nav: SEARCH · EVIDENCE · HOW IT WORKS · TEAM |
| **Kicker** | tiny red mono caps above a heading | `PRECEDENT & STATUTE SEARCH` |
| **Hero** | huge Archivo headline (part in red), short Inter paragraph, two buttons, a small link below; a line diagram on the right | headline idea: `EVERY JUDGMENT IS A STATION. / PRECEDENT IS THE TRAIL.`; diagram = 3 coloured lines (keyword / neural / graph) joining at "interchange" judgments |
| **Buttons** | primary: solid ink, mono caps, square; secondary: 2px ink outline; on dark: red solid | `SEARCH A JUDGMENT`, `SEE THE EVIDENCE` |
| **Ticker strip** | full-width band, mono caps metrics separated by red ◆, scrolling slowly | real numbers from `/results`: `◆ 237 TEST QUERIES ◆ µF1@K 0.xx ◆ 100 IPC SECTIONS ◆ STUDENT MODEL xx MB ◆ yy MS ON CPU` (respect `prefers-reduced-motion`: no scroll) |
| **Stat row** | 4 columns with mono label (`STOPS`, `HOURS`…) and big value, hairline separators | `RESULTS`, `MS`, `TOP STATUTE`, `AGREEMENT` |
| **Trail timeline** | vertical dashed line with ring "stations"; each item: Archivo caps title + small amber tag, mono metadata line, small outline button `WHY THIS STEP?` + `OPEN ↗` | each retrieved **judgment** is a station: title/case id, mono line `RANK 01 · BM25 12.4 · NEURAL 0.81 · PAGERANK 0.003`, button `WHY THIS CASE?` opens the signal breakdown + matched passages; the top match uses a filled/target ring, "interchange" diamond for cases shared by several signals |
| **Mechanism cards** | 4 columns, each with a coloured square number badge `01` and Archivo caps title, mono body text | `01 KEYWORD` (red, BM25), `02 MEANING` (blue, neural), `03 AUTHORITY` (green, PageRank/HITS), `04 STATUTES` (amber, classifier) |
| **Evidence tiles** | huge coloured mono number, small mono comparison line below (`baseline: …`), mono caps label | `µF1@K (OURS)` vs `39.15 IL-TUR BASELINE`, `MACRO-F1 (STATUTES)` vs `28.08 LeSICiN`, `MODEL SIZE`, `CPU LATENCY` |
| **Footer CTA** | ink band, big Archivo headline, red button | `MIND THE PRECEDENT.` + `SEARCH A JUDGMENT →` |
| **Statute chips** | square amber-outlined chips with mono text and a confidence bar | `IPC 304B · 0.91` |

## 6. Pages
1. **`/` Landing** — TopBar → Hero → Ticker → "A trail, plotted" (dark section with a real example query and its trail timeline) → "How the trail is built" (4 mechanism cards) → "Measured, not claimed" (evidence tiles) → Footer CTA.
2. **`/search`** — big textarea "Paste a judgment or describe the facts", example buttons (real test-set judgments), system selector (`BM25`, `+ GRAPH`, `+ NEURAL`, `FULL`). Output: stat row, trail timeline of top-10 precedents, statute chips with confidence, and a side panel **"What the engine did"**: tokens/query reduction, top terms with df/idf, PageRank of each result, neural score, fusion weights.
3. **`/evidence`** — all charts and tables from `results/` (ablation table, NB vs Rocchio vs kNN vs neural, quality vs model size/latency, comparison with published baselines), each with a one-line "How to read" and a "Takeaway" computed from the data.
4. **`/how-it-works`** — the pipeline diagram and one card per component, each naming the source file.
5. **`/team`** — three cards: name, roll number, what they built.

## 7. Interaction and quality rules
- Loading state: mono caps `PLOTTING THE TRAIL…`; error state: red mono message with the API error.
- Keyboard accessible; visible focus ring (2px `--line-red` outline); text contrast ≥ 4.5:1.
- Respect `prefers-reduced-motion` (stop the ticker, no animated line drawing).
- No lorem ipsum, no placeholder numbers, no stock photos.
- Keep components small (< ~150 lines); no business logic in the frontend.
