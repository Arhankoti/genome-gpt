# Fingerprints for DNA

*Part 14 of a series on teaching an AI to read DNA. Last post, the lesson was to ask two tools that work differently and trust them only when they agree. This post turns to a tool I've had since Part 1 and never checked: the embedding, which squashes any stretch of DNA into a list of 384 numbers. If those numbers are a good fingerprint, "find me something like this" becomes possible. If they're just GC content in a trench coat, I'd rather know before building on them.*

---

## What an embedding is

When my model reads a stretch of DNA, every position ends up with an internal vector, the model's working summary of what it has read so far. Average those vectors over a window and you get one list of 384 numbers for the whole window. That's `embed()`. Two windows with similar lists should, in theory, be similar sequences.

The suspicious part is the word "similar." In bacterial DNA, one number explains an enormous amount about what a sequence looks like: the fraction of letters that are G or C. It runs from 39% in *H. pylori* to 66% in *M. tuberculosis* among these genomes. Part 10's codon-style finding and Part 13's shadow results were both partly GC in disguise, so I expected the embedding to be too. The question is how much, and whether anything is left over.

## The experiment

Windows of 512 letters from all 20 genomes in the corpus (two of them, *M. tuberculosis* and *H. pylori*, were never seen in training). For each genome, 60 windows come from the first half of the genome (a "library") and 60 from the second half (the "queries"). So a query never overlaps its neighbours: 1,200 library windows, 1,200 queries.

**Test A, which genome is this from?** Each query looks up its most similar library window by cosine similarity, and I count how often the neighbour is from the right genome. Chance is 5%. It's a crude version of "find me something like this."

**Test B, how much of the embedding is G+C?** A linear probe predicts a window's GC from its embedding, trained on the library, scored on the queries.

**Test C, is there biology in there?** Inside each held-out genome, windows lying wholly inside a known gene are labelled by which strand the gene is on. A linear probe tries to tell + from −.

Next to every embedding number I ran the dumb version: GC alone, a count of all 256 four-letter words (a "4-mer spectrum"), each with the GC component linearly removed, and queries whose letters were shuffled (same composition, no order).

## Result 1: the embedding is mostly a GC meter

| Linear probe predicting a window's G+C (held-out R²) | |
|---|---|
| from the 4-mer counts | 1.00 |
| from the embedding | 0.96 |

R² of 1.00 for 4-mers isn't surprising, since GC is just a weighted sum of those counts. But the embedding gets 0.96. About 96% of what varies between windows in those 384 numbers can be read off as one quantity a child could count by hand.

## Result 2: as a fingerprint, it loses to counting

| Which of 20 genomes? (1-NN, chance 5%) | real queries | letter-shuffled queries |
|---|---|---|
| GC only | 5.0% | 5.0% |
| embedding (z-scored) | 16.4% | 11.2% |
| embedding, GC removed | 16.4% | 14.2% |
| **4-mer counts** | **51.7%** | 14.2% |
| 4-mer counts, GC removed | 46.5% | 10.7% |

([`fingerprints.png`](fingerprints.png) has the bars.) The embedding is above chance, but it identifies the source genome about 16% of the time while a plain letter-count identifies it 52% of the time. Removing GC barely changes either one, so the embedding's weak signal isn't GC. It's something else, just not much of it.

I gave the embedding every fair chance I could think of. Raw cosine, mean-centered, z-scored per dimension: 15%, 16%, 16%. They're all the same because the problem isn't how I measure distance. On the two held-out genomes the embedding is at 20% (about 120 queries, so treat that as a rough number), versus 55–69% for the 4-mer counts.

The shuffle control is the part that stings in a different way. When I shuffle each query's letters, 4-mer identification falls from 52% to 14%, so the 4-mer counts clearly use order. The embedding falls only from 16% to 11%. So most of the little it does know about a genome is available from composition alone, which isn't "nothing" but also isn't grammar.

## Result 3: strand is in there, and it's still no better than counting

| Coding strand, linear probe (accuracy) | *M. tuberculosis* | *H. pylori* |
|---|---|---|
| always guess the majority strand | 55% | 64% |
| GC | 44% | 36% |
| embedding, probe trained on shuffled labels | 54% | 48% |
| **4-mer counts** | 85% | **90%** |
| **embedding** | 85% | 82% |

(250 test windows per genome.) This is the good news in the post. From 512 letters inside a gene, a linear probe on the embedding tells which strand the gene is on, 85% and 82% of the time, far above the control rows. So the embedding does carry coding structure: it knows which direction the codons run. But 4-mer counts get there too: a tie in *M. tuberculosis*, and 8 points better in *H. pylori*.

That's the same story as Part 11: the model has learned the codon rhythm, but a counter that reads the same letters finds it as well. The embedding isn't empty, it just doesn't beat the baseline.

## The part where I tried to make a tool out of it, and measured the tool

Since `embed()` returns 384 raw floats (which is a lot of tokens for a frontier model to read and no use if it can't interpret them), I wanted a bounded tool that answers "are these two sequences alike?" So I built `dna_compare`. It returns the embedding cosine, the cosine against a *shuffled copy* of the second sequence (same letters, no order), the 4-mer cosine, and both GC values. The idea: if the real sequence is closer to the first one than its own shuffle is, that gap is the part order contributes.

Then I ran it on seven pairs of 512-letter windows from the real genomes.

| pair | cosine | cosine vs shuffled b | gap |
|---|---|---|---|
| *E. coli* vs *E. coli*, another locus | 0.981 | 0.998 | −0.017 |
| *E. coli* vs *Salmonella* | 0.987 | 0.991 | −0.005 |
| *E. coli* vs *Y. pestis* | 0.998 | 0.998 | 0.000 |
| *E. coli* vs *B. subtilis* | 0.980 | 0.995 | −0.015 |
| *E. coli* vs *M. tuberculosis* | 0.978 | 0.996 | −0.018 |
| *M. tb* vs *M. tb*, another locus | 0.995 | 0.997 | −0.002 |
| *M. tb* vs *P. aeruginosa* (both GC-rich) | 0.992 | 0.994 | −0.003 |

Every real window sits at 0.98–1.00 cosine to every other real window, related or not. And a letter-shuffled copy is, in all seven pairs, *at least as close* as the genuine sequence. Never closer to the real thing than to its own scrambled version. Mean-pooled embeddings of DNA-like text all land in nearly the same direction, and a shuffled sequence lands right in the middle of that crowd.

So the tool I built can't do the job I built it for. It can only report that there's "no evidence beyond composition," which is what it said seven times out of seven. I kept it, and the schema says it plainly, because a tool that reliably says "don't trust a similarity score from the embedding" is more useful to the big model than one that returns 0.99 and lets it conclude two sequences are related. But I don't expect anyone to find sequence similarity with it.

## What I built

- **`fingerprints.py`**: the pure helpers: 4-mer vectors, cosine and nearest-neighbour accuracy, ridge probes, a "remove this covariate linearly" step (used to take GC out of a representation), and `similarity_report`, which takes the model as a function so it stays model-free.
- **`neighbors.py --fingerprint_test`**: the experiment above, end to end. Every number in this post comes from one run with seed 0.
- **`dna_compare`**, the 13th bridge tool: bounded scalars only, with the verdict above.
- **`viz.render_fingerprints`**: the two-panel figure.

Things I didn't do, which matter: one window length (512), one seed, one embedding recipe (mean of the final layer). A different pooling or an intermediate layer might carry more genome identity; I haven't tried. And "which genome" is a blunt proxy for "similar": related species such as *E. coli* and *Salmonella* arguably *should* be confused.

## Where this goes next

The embedding is a weaker fingerprint than a letter count, and mostly a GC meter. That's a clear answer and it changes the plan: building "find me something like this" (similarity search) on top of raw embeddings would give me a search engine that returns GC-matched strangers. If I want similarity search, the honest first version is probably the 4-mer counts, with the neural embedding as the thing that has to beat them, perhaps by using a different layer or contrasting the model against itself.

Next, **Post 15**, picks that up: build the nearest-neighbour search properly (a reference library, a query, ranked hits), with the 4-mer baseline and a shuffle control in the loop from the start, and ask whether any variant of the embedding can add something the counts don't have.
