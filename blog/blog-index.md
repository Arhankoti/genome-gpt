# Genome-GPT: The Whole Journey (Blog Index)

*A living roadmap for the series. The rule for every entry is the same: **learn something, build something, publish something.** No post ships without a matching code increment, and no code increment ships without a post explaining what it taught me.*

This is a plan, not a promise. The near-term entries are concrete. The far ones are directions — real destinations, but I fully expect the map to change as the project teaches me what actually matters. That's the whole point.

**Legend:** ✅ published · 📝 written, not yet posted · 🔨 code exists · 🧭 planned · 💭 directional

---

## Arc I — Foundations: can a machine read DNA at all? (Posts 1–8)

The "why," the smallest possible model, and — most importantly — learning how to measure honestly so I don't fool myself.

1. ✅ **Life Is Written in Four Letters** — why any of this is worth caring about.
2. 📝 **Teaching a Baby to Read** — a tiny GPT, bits-per-base, and the Markov baseline that humbled me. *(Delivers on Post 1's closing promise: the training story, and how it "gets humbling fast.")*
3. 📝 **Which Letters Actually Matter** — saturation mutagenesis: scan every position, map the ones that break things. *(Part 3 — code landed (`saturation_scan` + `landscape.py` + `dna_saturation_scan`) and post drafted: [`blog-3-which-letters-actually-matter.md`](blog-3-which-letters-actually-matter.md).)*
4. 📝 **The Model Dreams** — what happens when you let it generate DNA from scratch, and how to tell if the dream is plausible (fidelity *and* novelty). *(Part 4 — code landed (`generation.py` + `dream.py` + `dna_generation_report`) and post drafted: [`blog-4-the-model-dreams.md`](blog-4-the-model-dreams.md).)*
5. 📝 **Real Bacteria, Real Grammar** — moving off toy data onto real genomes; where the neural net finally beats the counter (it does — biggest on high-GC *M. tuberculosis*, where the counter scores worse than random). *(Part 5 — code landed: whole-genome holdout + `benchmark.py`; post drafted: [`blog-5-real-bacteria-real-grammar.md`](blog-5-real-bacteria-real-grammar.md).)*
6. 📝 **How Big Is Big Enough?** — scaling the baby up and watching the honest number move — until the train–val gap fans open and bigger just means memorizing. *(Part 6 — code landed: `scaling.py` + `params_for_config`; post drafted: [`blog-6-how-big-is-big-enough.md`](blog-6-how-big-is-big-enough.md).)*
7. 📝 **The Data Is the Model** — cleaning, deduping, and the leakage guard: a near-duplicate strain across the train/test line would fake the Part 5 win (it doesn't — the corpus passes the check). *(Part 7 — code landed: `data/quality.py` MinHash dedup + `dataqc.py` + a leakage guard in `prepare.py`; post drafted: [`blog-7-the-data-is-the-model.md`](blog-7-the-data-is-the-model.md).)*
8. 💭 **Arc I Retrospective** — what a six-letter language taught me about learning itself.

## Arc II — Giving the model hands: tools (Posts 9–20)

A model that only outputs a number isn't useful yet. This arc turns it into a set of tools a person (or a bigger AI) can actually call.

9. 💭 **Scoring, Out Loud** — turning "bits per base" into a plain-English "does this look like real DNA?"
10. 💭 **The Variant Detective** — single-letter changes and estimating which ones are disruptive.
11. 💭 **Saturation Landscapes** — heatmaps of every possible mutation across a gene. *(Deep-dive on the Part 3 build.)*
12. 💭 **Old-School Biology Tools** — ORF finders, GC scans, restriction sites, sitting right next to the neural ones.
13. 💭 **Neural + Classical, Together** — when the dumb deterministic tool is right and the smart model is wrong.
14. 💭 **Fingerprints for DNA** — embeddings: turning a sequence into a vector.
15. 💭 **Find Me Something Like This** — similarity search over embeddings.
16. 💭 **A Contract, Not a Guess** — designing the tool schema so nothing silently breaks.
17–20. 💭 **Tooling polish** — batching, long-sequence handling, error cases, and making the tools trustworthy enough to build on.

## Arc III — The bridge: a big AI that calls my small one (Posts 21–34)

The original vision: a frontier model handles language and reasoning, and calls my specialist model for anything DNA-native. This arc builds that bridge for real.

21. 💭 **Two Brains, One Question** — why the big model should never reason over raw bases.
22. 💭 **My First Genomics Assistant** — a chat loop where the big model calls my tools.
23. 💭 **When the Big Model Asks the Wrong Thing** — debugging tool-call reasoning.
24–28. 💭 **Making the assistant reliable** — grounding answers, refusing to hallucinate biology, showing its work.
29–34. 💭 **A real workflow** — walking a genuine variant question end to end, from plain English to a defensible answer.

## Arc IV — Scaling the science (Posts 35–52)

Bigger context, better architecture, harder data. This is where the ML gets serious.

35–40. 💭 **Longer memory** — getting past the ~1,000-letter context limit so the model can see whole genes.
41–46. 💭 **Better architectures** — trying the ideas (byte-level, state-space blocks) that the toy model couldn't justify.
47–52. 💭 **From bacteria to bigger genomes** — what breaks when sequences get long and messy, and honestly, what I can afford to run.

## Arc V — Real biology questions (Posts 53–74)

The payoff arc. Pointing the whole system at questions that sound like the ones from Post 1.

53–58. 💭 **Where do genes start and stop?** — promoters, start codons, and whether the model learned them without being told.
59–64. 💭 **Comparing organisms** — what the model's "surprise" reveals about how related two genomes are.
65–70. 💭 **The needle in the haystack** — ranking which changes in a sequence are worth a human's attention.
71–74. 💭 **Honest limits** — a full post on everything this system *can't* do, and why that matters.

## Arc VI — Shipping, sharing, and the long tail (Posts 75–100)

Turning a bedroom project into something other people can use, poke at, and learn from.

75–80. 💭 **A fast serving tier** — porting inference to Rust so the tools respond instantly.
81–86. 💭 **A demo anyone can click** — a web page where a curious kid types DNA and sees the landscape light up.
87–92. 💭 **Open sourcing it properly** — docs, tests, and inviting other students in.
93–98. 💭 **What I got wrong** — a serial post-mortem of the dumbest and most instructive mistakes.
99. 💭 **What I'd Tell Myself at Post 1.**
100. 💭 **Where DNA-reading AI Goes Next** — the honest state of the real field, and where a next person could start.

---

## How the index stays honest

- Every ✅/📝 entry links to the exact code increment that produced it.
- Entries move from 💭 → 🧭 → 🔨 → 📝 → ✅ as they get real. Nothing jumps straight to published.
- Renumbering is allowed and expected. If Post 40's idea turns out to matter more than Post 12's, it moves up. The arc structure is the fixed part; the exact titles are not.
- If an arc collapses (an idea doesn't pan out), that failure becomes its own post rather than getting quietly deleted.

*Current status: Post 1 published; Post 2 drafted; Post 3 code landed (plan in [`plans/part-3-saturation-mutagenesis.md`](../plans/part-3-saturation-mutagenesis.md)) and post drafted; Post 4 code landed (plan in [`plans/part-4-the-model-dreams.md`](../plans/part-4-the-model-dreams.md)) and post drafted; Post 5 code landed (plan in [`plans/part-5-real-bacteria-real-grammar.md`](../plans/part-5-real-bacteria-real-grammar.md)) and post drafted — first real-data win; Post 6 code landed (plan in [`plans/part-6-how-big-is-big-enough.md`](../plans/part-6-how-big-is-big-enough.md)) and post drafted — scaling curve + train–val gap; Post 7 code landed (plan in [`plans/part-7-the-data-is-the-model.md`](../plans/part-7-the-data-is-the-model.md)) and post drafted — MinHash dedup + leakage guard; the Part 5 corpus passes the check.*
