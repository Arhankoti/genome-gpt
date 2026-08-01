# Teaching a Baby to Read

*Part 2 of a series on teaching an AI to read DNA. Part 1 was the daydream. This is where I actually opened the laptop, wrote the code, and got humbled.*

---

In Part 1 I made a promise. I said the next post would be about how you take a few million letters of DNA and teach a tiny "baby" language model to find the patterns hiding inside them. I also warned you it gets humbling fast.

I did not know how fast.

Here is the honest version of what happened when I stopped daydreaming and started typing.

## The plan was almost stupidly simple

If DNA is a language written in four letters, and a language model learns a language by guessing the next symbol, then the recipe writes itself.

Take a genome. It's just a giant text file of `A`, `C`, `G`, and `T`. Show a model a chunk of it. Hide the next letter. Ask it to guess. Tell it whether it was right. Do that a few million times.

That's it. That's the whole idea. The same trick ChatGPT uses on English sentences, pointed at DNA instead. No fancy biology. Just "predict the next letter," over and over, until the model stops being random and starts having opinions about what usually comes next.

So I built the smallest version of that I could. A little GPT, the same shape as the ones that power chatbots, just shrunk way down. Mine has about 12 million knobs it can turn. For comparison, the big models have hundreds of billions. This is a baby. That's the point.

There isn't even a real vocabulary to learn. English models have to deal with tens of thousands of possible tokens. Mine has six: `A`, `C`, `G`, `T`, `N` (which means "unknown letter"), and one special symbol I'll explain later. Six. You could write the whole dictionary on a Post-it.

## The first run felt amazing (this was the trap)

I grabbed a single genome — *E. coli*, the famous gut bacterium, a free download — and started training.

And it *worked*. The number I was watching, the model's error, dropped and dropped. The thing was learning. I sat there watching a graph go down and genuinely felt like a scientist.

Then a quiet, annoying thought showed up.

*How do you know it learned anything real? What if it's just memorizing?*

This is the trap, and it's worth slowing down on, because it's the exact moment the project stopped being cute and started teaching me something.

A model with millions of knobs is perfectly capable of just... memorizing the answer sheet. If I train it on one genome and then test it on that same genome, of course it looks smart. It has basically seen the test before. That's not reading. That's a kid who memorized the eye chart.

I needed a way to tell the difference between *learning the language* and *memorizing the page*.

## The one number that keeps you honest

The tool for this turns out to be a single number, and I've come to love it.

You measure the model in **bits per base**: how surprised it is, on average, by each new letter. Think of it as "how many yes/no questions would you need to guess this letter."

If the model is totally clueless and every letter is a coin-flip among four, that's **2.0 bits**. That's the "you learned nothing" line. Pure randomness.

If the model has actually picked up on real patterns in DNA, it should be *less* surprised than that. Below 2.0. The lower the number, the more the model "gets it."

So now I had a scoreboard. Below 2.0, good. At 2.0, you're a random number generator wearing a lab coat.

But there was still a hole in this. Beating "random" is a low bar. Random is *nothing*. If I want to claim my neural network learned something about the grammar of life, it should have to beat something that actually tries.

## Enter the dumbest possible opponent

So I built a rival on purpose, to keep myself honest. The nerdy name is a **Markov model**. The plain version: it's just counting.

You go through the genome and tally up, "after I see `ATG`, what letter comes next, and how often?" Then to make a prediction, you just look up your tally. No neural network. No training in the fancy sense. No mystery. It's a program a motivated person could write in an afternoon, and honestly it barely deserves to be called AI.

That counter is the bar. My whole fancy neural network has one job: **beat the counter.** If a 12-million-knob transformer can't out-predict a lookup table, then it has justified nothing, and I should go do my actual homework.

Setting up that fight is the single most useful thing I did. Not the model. The *opponent*.

## The humbling, in three parts

Part one of the humbling: the counter is *good*. Frustratingly good. DNA has a lot of local, short-range regularity, and simple counting soaks up a shocking amount of it. My neural net did not get to feel special. It had to claw for every fraction of a bit against a program with no brain.

Part two was funnier and more instructive. I let the counter cheat by looking at longer and longer context — count what follows every 8-letter combo instead of every 3-letter one. You'd think more context is always better. Instead it fell apart. On letters it had trained on it looked like a genius, but on *new* letters it had never seen, it got worse than random — I clocked one version at **2.40 bits**, which is worse than just guessing.

That's memorization, caught red-handed. The counter had basically made flashcards for every 8-letter string in the training data and had nothing left over for anything new. Watching a model score *worse than a coin flip* because it over-memorized was the clearest lesson in the whole project. It's the eye-chart kid again, failing the moment you swap the chart.

Part three was the mirror. Because everything the counter did wrong, my neural net could do too, just more sneakily. The whole reason bits-per-base and the Markov opponent matter is that they're the only things standing between me and fooling myself.

## What being humbled actually changed

Getting beaten up by a lookup table forced four fixes, and each one is really a lesson in disguise.

**Test on genomes it has never seen.** One genome isn't enough — the model can just memorize it. So I switched to training on many different bacteria and holding a few of them *completely out* of training. The model only earns its score on organisms it has never laid eyes on. That's the difference between recall and reading, and it's the whole ballgame.

**Don't let it learn lies at the seams.** When you glue many genomes into one long file, the model will happily "learn" that the end of one genome flows into the start of the next — a pattern that means nothing, because those two organisms have never met. That's what that mysterious sixth symbol is for. It's a little fence post I plant between genomes so the model knows *this is a boundary, don't read across it.*

**Teach it that DNA has two sides.** DNA is double-stranded, and the two strands are mirror images with the letters swapped in a fixed way (`A` pairs with `T`, `C` with `G`). A pattern is just as real read forwards on one strand as backwards-and-swapped on the other. So half the time, I flip the sequence to its mirror-image strand during training. Same biology, seen from the other side. Free extra data, and it nudges the model toward real structure instead of memorized quirks.

**Trust the scoreboard over the vibes.** The graph going down felt amazing and meant almost nothing on its own. The number that matters is the one measured on data the model never trained on, compared against the dumbest opponent I could build. Feelings lie. The held-out bits-per-base doesn't.

## Where this leaves me (and the honest part)

I want to be straight about what I have and haven't proven, because Part 1 was about *why* and I don't want Part 2 to quietly turn into hype.

What I've built so far is mostly the **referee**, not the champion. The harness. The measuring stick, the honest opponent, the held-out test, the fences between genomes. And when I run the whole thing on carefully controlled *toy* data where the different genomes deliberately share no grammar, nothing beats random — which is exactly correct, because there's nothing real to learn, and it means my ruler isn't lying to me.

I should be precise about one thing, because it's the difference between a blog and a brag: that k=8 counter melting down to 2.40 bits while my neural net held near 2.0? That was on exactly this controlled toy data. It's a clean demonstration that my *referee* works — that I can catch memorization when it happens — not a headline that my model has cracked real biology. I haven't earned that sentence yet. I've only earned the right to measure honestly when I go looking for it.

The actual prize is still ahead: real bacteria genuinely *do* share grammar, deep patterns that a lookup table can't fully capture but a neural net might. The gap between "what counting can do" and "what the model can do" on real, shared biology — closing that gap is the entire point. That's the thing I'm chasing now.

Blog 1 was the invitation. Blog 2 is the part where the universe checks whether you were serious. It turns out most of learning to build this was really learning how to *not fool myself* — because a model with millions of knobs will absolutely tell you what you want to hear if you let it.

That's Part 3: pointing this thing at a real question. Now that I can measure honestly, the next move is to make the model do something a curious kid can actually *look at* — take a stretch of DNA and ask, letter by letter, *which of these actually matter?* Which single changes would break it, and which the model just shrugs at.

That question is where Part 1 started. "The letters actually matter." Next post, I try to get the model to show me which ones.
