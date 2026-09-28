"""Render analysis outputs (saturation scans, generation sweeps) as PNGs.

Matplotlib is an optional, dev-only dependency: it is imported lazily inside each
function so importing this module (or the rest of the package) never requires it.
Inputs are plain dicts (from GenomeModel.saturation_scan / generation.dream_report)
— this module has no coupling to the model itself.
"""

# alt-base row order matches the grid columns produced by saturation_scan.
ALT_BASES = ("A", "C", "G", "T")


def render_landscape(scan, out_path="landscape.png", title=None):
    """Heatmap of a saturation scan: x = position, y = alt base, color = LLR.

    Uses a diverging colormap centered at 0 (red = disruptive/negative LLR, blue
    = tolerated/positive). The single most-disruptive cell is annotated. Axes are
    labeled with absolute positions.

    Args:
        scan: The dict returned by GenomeModel.saturation_scan. Must contain
            "grid", "positions", "ref_bases", and "most_disruptive".
        out_path: Where to write the PNG.
        title: Optional figure title.

    Returns:
        The out_path written.

    Raises:
        ImportError: If matplotlib is not installed (it is a dev-only dependency).
        ValueError: If the scan has no scored positions to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_landscape needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    grid = scan["grid"]
    positions = scan["positions"]
    if not grid:
        raise ValueError("scan has no scored positions to plot (n_scored == 0)")

    # grid is (n_scored, 4); transpose to (4 alt bases, n_scored) for the heatmap.
    data = np.asarray(grid, dtype=float).T
    vmax = float(np.abs(data).max()) or 1.0  # guard the all-zero (untrained) case
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    fig_w = max(6.0, min(24.0, len(positions) * 0.06))
    fig, ax = plt.subplots(figsize=(fig_w, 2.6))
    im = ax.imshow(data, aspect="auto", cmap="RdBu", norm=norm, interpolation="nearest")

    ax.set_yticks(range(len(ALT_BASES)))
    ax.set_yticklabels(ALT_BASES)
    ax.set_ylabel("alt base")
    ax.set_xlabel("position (bp)")

    # Thin the x ticks so long sequences stay readable; label with absolute pos.
    n = len(positions)
    step = max(1, n // 20)
    xticks = list(range(0, n, step))
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(positions[i]) for i in xticks], rotation=90, fontsize=7)

    # Annotate the single most-disruptive substitution.
    if scan.get("most_disruptive"):
        top = scan["most_disruptive"][0]
        if top["position"] in positions:
            xi = positions.index(top["position"])
            yi = ALT_BASES.index(top["alt_base"])
            ax.scatter([xi], [yi], marker="o", s=60, facecolors="none", edgecolors="black")
            ax.annotate(
                f"{top['ref_base']}{top['position']}{top['alt_base']} (LLR {top['llr']:.2f})",
                xy=(xi, yi),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=7,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("log-likelihood ratio (alt − ref)")

    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_dream_sweep(report, out_path="dream_sweep.png", title=None):
    """Plot the fidelity-vs-novelty tension across sampling temperature.

    Two curves share the temperature x-axis:
        * fidelity — kmer_js_bits (lower = more DNA-like), left y-axis.
        * novelty  — copied_kmer_fraction (lower = less plagiarized), right y-axis.
    The 'sweet spot' temperature (good fidelity AND low copying) is shaded and
    annotated. Input is exactly the dict from generation.dream_report — no model
    coupling.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If the report has no sweep rows.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_dream_sweep needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    sweep = report.get("sweep", [])
    if not sweep:
        raise ValueError("report has no sweep rows to plot")

    temps = [row["temperature"] for row in sweep]
    fidelity = [row["kmer_js_bits"] for row in sweep]  # lower = better
    novelty = [row["copied_kmer_fraction"] for row in sweep]  # lower = better

    # Sweet spot: rank each temperature by fidelity + copying (both lower-better)
    # on a 0-1 normalized scale and pick the minimum combined score.
    def _norm(xs):
        lo, hi = min(xs), max(xs)
        rng = hi - lo
        return [0.0 for _ in xs] if rng == 0 else [(x - lo) / rng for x in xs]

    combined = [f + c for f, c in zip(_norm(fidelity), _norm(novelty))]
    best_i = min(range(len(combined)), key=lambda i: combined[i])

    fig, ax1 = plt.subplots(figsize=(7.0, 4.2))
    color_f, color_n = "tab:blue", "tab:red"

    ln1 = ax1.plot(temps, fidelity, "o-", color=color_f, label="fidelity: k-mer JS (bits)")
    ax1.set_xlabel("sampling temperature")
    ax1.set_ylabel("k-mer JS divergence (bits) — lower = more DNA-like", color=color_f)
    ax1.tick_params(axis="y", labelcolor=color_f)

    ax2 = ax1.twinx()
    ln2 = ax2.plot(temps, novelty, "s--", color=color_n, label="novelty: copied-k-mer fraction")
    ax2.set_ylabel("copied-k-mer fraction — lower = less plagiarized", color=color_n)
    ax2.tick_params(axis="y", labelcolor=color_n)
    ax2.set_ylim(-0.02, 1.02)

    # shade + annotate the sweet spot
    ax1.axvspan(temps[best_i] - 0.03, temps[best_i] + 0.03, color="green", alpha=0.12, zorder=0)
    ax1.annotate(
        f"sweet spot\nT={temps[best_i]:g}",
        xy=(temps[best_i], fidelity[best_i]),
        xytext=(6, 12),
        textcoords="offset points",
        fontsize=8,
        color="green",
    )

    lns = ln1 + ln2
    ax1.legend(lns, [line.get_label() for line in lns], loc="upper center", fontsize=8)
    ax1.set_title(title or "Generation sweep: fidelity vs novelty across temperature")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# Two-series categorical palette (neural vs Markov). Blue/orange is CVD-safe
# (validated: adjacent ΔE ~25); direct value labels supply the contrast relief.
_NEURAL_COLOR = "#1f77b4"
_MARKOV_COLOR = "#ff7f0e"


def render_benchmark(report, out_path="benchmark.png", title=None):
    """Grouped bars per held-out genome: neural bits/bp vs best-Markov bits/bp.

    Both series share ONE y-axis (both are bits/bp), the 2.0 'random' line is
    marked, and genomes are sorted by gap so the win/loss story reads left to
    right (neural-wins first). Input is exactly the dict from benchmark.benchmark.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If the report has no per-genome rows to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_benchmark needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    rows = sorted(report.get("per_genome", []), key=lambda r: r["gap_vs_best_markov"])
    if not rows:
        raise ValueError("report has no per_genome rows to plot")

    names = [r["name"] for r in rows]
    neural = [r["neural_bits_per_bp"] for r in rows]
    markov = [r["best_markov_bits_per_bp"] for r in rows]
    x = np.arange(len(names))
    w = 0.4

    fig_w = max(6.0, min(20.0, len(names) * 1.1))
    fig, ax = plt.subplots(figsize=(fig_w, 4.2))
    b1 = ax.bar(x - w / 2, neural, w, label="neural GPT", color=_NEURAL_COLOR)
    b2 = ax.bar(
        x + w / 2,
        markov,
        w,
        label="best Markov k-gram",
        color=_MARKOV_COLOR,
    )

    # 2.0 = random over 4 bases — the "learned nothing" line
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax.annotate(
        "random (2.0)",
        xy=(0, 2.0),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )

    # direct value labels (also the contrast relief for the orange series)
    ax.bar_label(b1, fmt="%.3f", fontsize=6, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=6, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("bits per base — lower = more predictable")
    # focus the y-range on where the bars actually live so small gaps are visible
    lo = min(min(neural), min(markov))
    hi = max(max(neural), max(markov), 2.0)
    ax.set_ylim(max(0.0, lo - 0.05), hi + 0.08)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9", linewidth=0.6)
    ax.set_title(title or "Neural vs Markov on held-out genomes (lower = better)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# val vs train share ONE bits/bp axis; the gap between them is the shaded band
# (memorization signal), never a second y-scale.
_VAL_COLOR = "#1f77b4"
_TRAIN_COLOR = "#ff7f0e"


def render_scaling(report, out_path="scaling.png", title=None):
    """Val and train bits/bp vs parameter count (log x, shared bits/bp y-axis).

    The shaded band between the curves is the train-val gap; where val stops
    improving while the band fans open is the model going from learning to
    memorizing. The 2.0 random line is marked. Input is exactly the dict from
    scaling.run_ladder.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If the report has fewer than one rung.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_scaling needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    rungs = report.get("rungs", [])
    if not rungs:
        raise ValueError("report has no rungs to plot")

    params = [r["params"] for r in rungs]
    val = [r["val_bits_per_bp"] for r in rungs]
    train = [r["train_bits_per_bp"] for r in rungs]

    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    ax.fill_between(params, train, val, color="gray", alpha=0.12, label="train–val gap")
    ax.plot(params, val, "o-", color=_VAL_COLOR, label="held-out (val) bits/bp — the honest number")
    ax.plot(params, train, "s--", color=_TRAIN_COLOR, label="train bits/bp")

    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    ax.annotate(
        "random (2.0)",
        xy=(params[0], 2.0),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )

    ax.set_xscale("log")
    ax.set_xlabel("non-embedding parameters (log scale)")
    ax.set_ylabel("bits per base — lower = better")
    # direct-label each rung so identity isn't size-only
    for r in rungs:
        ax.annotate(
            r["label"],
            xy=(r["params"], r["val_bits_per_bp"]),
            xytext=(0, -12),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=_VAL_COLOR,
        )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_axisbelow(True)
    ax.grid(color="0.92", linewidth=0.6)
    ax.set_title(title or "How big is big enough? Held-out bits/bp vs model size")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# One CVD-safe hue per corpus (validated blue/orange pair, adjacent ΔE ~25). The
# category here is the corpus, not val-vs-train, so color encodes which dataset.
_CORPUS_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd")


def render_scaling_compare(reports, labels, out_path="scaling_data.png", title=None):
    """Overlay several ladders that differ ONLY in training data (Part 8).

    Two panels, each a single bits/bp y-axis vs parameter count (log x):
        * left  — held-out (val) bits/bp per corpus: does the whole honest curve
          shift down when the corpus grows?
        * right — the train-val gap per corpus: does more data shrink the
          memorization gap at fixed capacity?
    One hue per corpus. The 2.0 random line is marked on the left panel. Inputs are
    exactly the dicts from scaling.run_ladder (aligned by scaling.align_runs).

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If fewer than one shared rung exists across reports.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_scaling_compare needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    from scaling import align_runs  # pure alignment core (shared with the CLI)

    aligned = align_runs(reports, labels)
    rungs = aligned["rungs"]
    params = [r["params"] for r in rungs]

    fig, (axv, axg) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for k, lab in enumerate(aligned["labels"]):
        color = _CORPUS_COLORS[k % len(_CORPUS_COLORS)]
        val = [r["val"][k] for r in rungs]
        gap = [r["gap"][k] for r in rungs]
        axv.plot(params, val, "o-", color=color, label=lab)
        axg.plot(params, gap, "o-", color=color, label=lab)

    axv.axhline(2.0, color="gray", linestyle=":", linewidth=1)
    axv.annotate(
        "random (2.0)",
        xy=(params[0], 2.0),
        xytext=(2, 3),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )
    axg.axhline(0.0, color="gray", linestyle=":", linewidth=1)

    for ax in (axv, axg):
        ax.set_xscale("log")
        ax.set_xlabel("non-embedding parameters (log scale)")
        ax.set_axisbelow(True)
        ax.grid(color="0.92", linewidth=0.6)
        ax.legend(loc="best", fontsize=8)
        # direct-label each rung so identity isn't size-only
        for r in rungs:
            ax.annotate(
                r["label"],
                xy=(r["params"], r["val"][0] if ax is axv else r["gap"][0]),
                xytext=(0, -12),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="0.4",
            )
    axv.set_ylabel("held-out (val) bits/bp — the honest number, lower = better")
    axg.set_ylabel("train–val gap (bits/bp) — higher = more memorization")
    axv.set_title("Does more data move the honest number?")
    axg.set_title("Does more data shrink the memorization gap?")
    fig.suptitle(title or "Scaling the corpus, not the model (ladder + budget frozen)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# One color per verdict category (categorical, CVD-safe-ish; markers + text labels
# carry the identity so it never rests on color alone).
_VERDICT_COLORS = {
    "dna_like": "#2ca02c",
    "plausible_composition": "#1f77b4",
    "random_like": "#7f7f7f",
    "low_complexity": "#ff7f0e",
}


def render_score_report(reports, out_path="score_verdicts.png", title=None):
    """A bits/bp 'thermometer': place scored sequences on ONE axis from natural to
    random, colored by verdict.

    x = neural bits/bp (lower = more natural), a single shared axis; the 2.0 random
    line and a shaded 'natural band' (clearly below random) are marked. Each input is
    a generation.score_verdict dict plus a "label"; points are drawn at their
    neural_bits_per_bp, colored by verdict, and annotated with the value + verdict so
    identity never rests on color alone.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If there are no reports to plot.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_score_report needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    if not reports:
        raise ValueError("no score reports to plot")

    rows = list(reports)
    ys = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.0, max(2.2, 0.7 * len(rows) + 1.2)))

    # natural band: clearly below random (the region a "dna_like" verdict lives in)
    ax.axvspan(1.5, 2.0 - 0.05, color="#2ca02c", alpha=0.06, zorder=0)
    ax.axvline(2.0, color="gray", linestyle=":", linewidth=1)
    ax.annotate(
        "random (2.0)",
        xy=(2.0, ys[-1]),
        xytext=(3, 6),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )

    for y, r in zip(ys, rows):
        x = r["neural_bits_per_bp"]
        color = _VERDICT_COLORS.get(r["verdict"], "#333333")
        ax.scatter([x], [y], s=90, color=color, zorder=3, edgecolors="white", linewidths=0.8)
        ax.annotate(
            f"{x:.3f} · {r['verdict']}",
            xy=(x, y),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=8,
            color=color,
        )

    ax.set_yticks(ys)
    ax.set_yticklabels([r.get("label", f"seq {i}") for i, r in enumerate(rows)], fontsize=8)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.invert_yaxis()
    # keep the interesting region in view but always include 2.0
    xmin = min(r["neural_bits_per_bp"] for r in rows)
    ax.set_xlim(min(1.75, xmin - 0.1), 2.06)
    ax.set_xlabel("neural bits per base — lower = more natural (2.0 = random)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="0.92", linewidth=0.6)
    ax.set_title(title or "Scoring, out loud: where each sequence lands, and the verdict")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_similarity(report, out_path="qc_similarity.png", title=None):
    """Genome x genome MinHash-Jaccard heatmap.

    Similarity is a magnitude with no meaningful midpoint, so it uses a SEQUENTIAL
    single-hue colormap (light = distinct -> dark = identical), not a diverging
    one. Cells at/above the dedup threshold are annotated — near-duplicate strains
    show up as hot off-diagonal blocks. Input is the dict from quality.qc_report.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If the report has no similarity matrix.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_similarity needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    sim = report.get("similarity", {})
    names = sim.get("names", [])
    matrix = sim.get("matrix", [])
    if not names or not matrix:
        raise ValueError("report has no similarity matrix to plot")

    data = np.asarray(matrix, dtype=float)
    threshold = report.get("dedup", {}).get("threshold", 0.9)

    n = len(names)
    fig_side = max(4.0, min(14.0, n * 0.7))
    fig, ax = plt.subplots(figsize=(fig_side, fig_side * 0.85))
    im = ax.imshow(data, cmap="Blues", vmin=0.0, vmax=1.0, interpolation="nearest")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(names, fontsize=7)

    # annotate near-duplicate cells (off-diagonal, >= threshold) so identity isn't
    # color-alone — the contrast relief the palette needs
    for i in range(n):
        for j in range(n):
            if i != j and data[i, j] >= threshold:
                ax.text(
                    j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white"
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("MinHash-Jaccard similarity (0 = distinct, 1 = identical)")
    ax.set_title(title or "Genome similarity — near-duplicates are hot blocks")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_DETECTIVE_COLORS = {"llr_neural": "#2a78d6", "llr_markov": "#eb6834"}
_DETECTIVE_LABELS = {"llr_neural": "neural (whole window)", "llr_markov": "k-gram baseline"}


def render_variant_classes(result, out_path="variant_classes.png", title=None):
    """The codon test in two panels (Part 10).

    Left: mean LLR per annotated variant class (synonymous / missense / nonsense),
    ±95% CI, for the neural model and the k-gram baseline on ONE shared nats axis.
    Right: separation AUC per class pair (P(first class scores lower)), for all
    variants and the GC-neutral subset, against the 0.5 can't-tell line. Values are
    labelled so identity never rests on color alone.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If there are no scored variants.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_variant_classes needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e
    import numpy as np

    rows = result.get("rows", [])
    if not rows:
        raise ValueError("no scored variants to plot")
    scorers = [s for s in ("llr_neural", "llr_markov") if s in rows[0]]
    classes = ("synonymous", "missense", "nonsense")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.2), gridspec_kw={"wspace": 0.45})

    # --- left: mean LLR per class, ±95% CI ---
    off = {s: (i - (len(scorers) - 1) / 2) * 0.22 for i, s in enumerate(scorers)}
    for s in scorers:
        for y, cls in enumerate(classes):
            vals = np.array([r[s] for r in rows if r["klass"] == cls], dtype=float)
            if vals.size == 0:
                continue
            m = vals.mean()
            ci = 1.96 * vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else 0.0
            yy = y + off[s]
            ax1.errorbar(m, yy, xerr=ci, fmt="o", color=_DETECTIVE_COLORS[s], ms=7, lw=2, capsize=0)
            ax1.annotate(
                f"{m:+.2f}",
                xy=(m, yy),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color="0.3",
            )
    ax1.axvline(0, color="gray", linestyle=":", linewidth=1)
    ax1.set_yticks(range(len(classes)))
    ax1.set_yticklabels(
        [f"{c}\n(n={sum(r['klass'] == c for r in rows)})" for c in classes], fontsize=8
    )
    ax1.set_ylim(-0.6, len(classes) - 0.4)
    ax1.invert_yaxis()
    ax1.set_xlabel("mean LLR (nats) — more negative = more disruptive")
    ax1.set_title("Mean LLR by annotated class", fontsize=10)
    ax1.grid(axis="x", color="0.92", linewidth=0.6)
    ax1.set_axisbelow(True)

    # --- right: separation AUC per class pair ---
    auc = result["summary"]["auc"]
    labels, ys = [], []
    y = 0
    for subset in ("all", "gc_neutral"):
        for pair in auc[subset]:
            a = auc[subset][pair]
            for s in scorers:
                v = a.get(s)
                if v is None or not np.isfinite(v):
                    continue
                ax2.scatter([v], [y + off[s]], s=45, color=_DETECTIVE_COLORS[s], zorder=3)
                ax2.annotate(
                    f"{v:.2f}",
                    xy=(v, y + off[s]),
                    xytext=(6, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=7,
                    color="0.3",
                )
            n = a.get("n", ["?", "?"])
            first, second = pair.split("_vs_")
            tag = " (GC-neutral)" if subset == "gc_neutral" else ""
            labels.append(f"{first} < {second}{tag}\nn={n[0]} vs {n[1]}")
            ys.append(y)
            y += 1
    ax2.axvline(0.5, color="gray", linestyle=":", linewidth=1)
    ax2.annotate(
        "can't tell (0.5)",
        xy=(0.5, -0.5),
        xytext=(3, 0),
        textcoords="offset points",
        fontsize=7,
        color="gray",
    )
    ax2.set_yticks(ys)
    ax2.set_yticklabels(labels, fontsize=7)
    ax2.set_ylim(-0.7, len(ys) - 0.4)
    ax2.invert_yaxis()
    ax2.set_xlim(0.3, 1.0)
    ax2.set_xlabel("separation AUC — P(first class scores lower)")
    ax2.set_title("Can the score tell the classes apart?", fontsize=10)
    ax2.grid(axis="x", color="0.92", linewidth=0.6)
    ax2.set_axisbelow(True)

    handles = [
        plt.Line2D(
            [], [], marker="o", ls="", color=_DETECTIVE_COLORS[s], label=_DETECTIVE_LABELS[s]
        )
        for s in scorers
    ]
    fig.legend(handles=handles, loc="lower center", ncol=len(scorers), frameon=False, fontsize=8)
    meta = result.get("meta", {})
    fig.suptitle(
        title
        or f"The codon test: single-letter changes in held-out {meta.get('genome', 'genome')} genes",
        fontsize=11,
    )
    fig.subplots_adjust(bottom=0.2, top=0.86)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_SHIFT_COLORS = {"S>W": "#2a78d6", "W>S": "#eb6834"}
_SHIFT_LABELS = {"S>W": "G/C → A/T", "W>S": "A/T → G/C"}


def render_codon_style(results, labels, out_path="codon_style.png", title=None):
    """Mean LLR by codon position, split by the direction of the base change (Part 10).

    One panel per genome (small multiples, shared nats axis). Filled markers are the
    neural model, hollow the k-gram baseline; color is the change direction
    (G/C->A/T vs A/T->G/C). A model that has learned the genome's codon-position
    composition shows a big, position-specific asymmetry that flips with the
    genome's GC content — its "writing style", not protein function.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If there are no results, or labels don't match.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_codon_style needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    if not results:
        raise ValueError("no codon-test results to plot")
    if len(labels) != len(results):
        raise ValueError("need one label per result")

    fig, axes = plt.subplots(
        1, len(results), figsize=(4.6 * len(results), 3.4), sharex=True, squeeze=False
    )
    vals = []
    for ax, res, label in zip(axes[0], results, labels):
        shift = res["summary"]["gc_shift"]
        for cp in ("1", "2", "3"):
            y = int(cp) - 1
            for j, d in enumerate(("S>W", "W>S")):
                c = shift[cp][d]
                yy = y + (j - 0.5) * 0.3
                color = _SHIFT_COLORS[d]
                for s, filled in (("llr_neural", True), ("llr_markov", False)):
                    v = c.get(s)
                    if v is None or v != v:  # missing or NaN
                        continue
                    vals.append(v)
                    ax.scatter(
                        [v],
                        [yy],
                        s=55,
                        zorder=3,
                        color=color if filled else "white",
                        edgecolors=color,
                        linewidths=1.6,
                    )
                ax.annotate(
                    f"{c['llr_neural']:+.2f}",
                    xy=(c["llr_neural"], yy),
                    xytext=(0, 6),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color="0.3",
                )
        gc = res.get("meta", {}).get("gc")
        ax.set_title(label + (f" (GC {gc:.0%})" if gc else ""), fontsize=10)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(["codon pos 1", "codon pos 2", "codon pos 3"], fontsize=8)
        ax.set_ylim(-0.6, 2.6)
        ax.invert_yaxis()
        ax.set_xlabel("mean LLR (nats) — negative = model dislikes it", fontsize=8)
        ax.grid(axis="x", color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    pad = 0.15 * (hi - lo or 1.0)
    axes[0][0].set_xlim(lo - pad, hi + pad)

    handles = [
        plt.Line2D([], [], marker="o", ls="", color=_SHIFT_COLORS[d], label=_SHIFT_LABELS[d])
        for d in ("S>W", "W>S")
    ] + [
        plt.Line2D([], [], marker="o", ls="", color="0.35", label="neural"),
        plt.Line2D(
            [], [], marker="o", ls="", markerfacecolor="white", color="0.35", label="k-gram"
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.suptitle(
        title or "What the model actually learned: each genome's codon-position style",
        fontsize=11,
    )
    fig.subplots_adjust(bottom=0.27, top=0.83, wspace=0.35)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


# fixed categorical order (slots 1-4); the shuffled control is a neutral gray
_FRAME_SERIES = (
    ("neural_expected_gc", "neural anticipation (expected G+C)", "#2a78d6", "-"),
    ("kgram_expected_gc", "k-gram anticipation (expected G+C)", "#eb6834", "-"),
    ("neural_landscape", "neural saturation landscape", "#1baf7a", "-"),
    ("gc_frame_plot", "classic GC frame plot", "#eda100", "-"),
    ("shuffled_neural_expected_gc", "control: shuffled window", "#8a8985", "--"),
)


def render_frame_accuracy(results, labels, out_path="frame_accuracy.png", title=None):
    """Reading-frame calling accuracy vs window length, one panel per genome (Part 11).

    x = window length (bp, log scale), y = fraction of held-out gene windows whose
    frame was called correctly (chance = 1/3, marked). One line per signal, direct-
    labelled at its right end so identity never rests on color alone.

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If there are no results, or labels don't match.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_frame_accuracy needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    if not results:
        raise ValueError("no frame-test results to plot")
    if len(labels) != len(results):
        raise ValueError("need one label per result")

    fig, axes = plt.subplots(
        1, len(results), figsize=(5.4 * len(results), 4.0), sharey=True, squeeze=False
    )
    for ax, res, label in zip(axes[0], results, labels):
        Ls = res["lengths"]
        ends = []
        for key, name, color, ls in _FRAME_SERIES:
            sig = res["signals"].get(key)
            if sig is None:
                continue
            ys = [sig["by_length"][str(L)]["accuracy"] for L in Ls]
            ax.plot(Ls, ys, color=color, linestyle=ls, linewidth=2, marker="o", markersize=5)
            ends.append([ys[-1], ys[-1]])
        # end labels: push apart so converging lines don't print on top of each other
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][1] = max(ends[i][1], ends[i - 1][1] + 0.045)
        for y, label_y in ends:
            ax.annotate(
                f"{y:.2f}",
                xy=(Ls[-1], y),
                xytext=(Ls[-1] * 1.07, label_y),
                textcoords="data",
                va="center",
                fontsize=7,
                color="0.3",
            )
        ax.axhline(1 / 3, color="gray", linestyle=":", linewidth=1)
        ax.annotate(
            "chance (1/3)",
            xy=(Ls[0], 1 / 3),
            xytext=(0, 4),
            textcoords="offset points",
            fontsize=7,
            color="gray",
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(Ls)
        ax.set_xticklabels([str(L) for L in Ls])
        ax.set_xlim(Ls[0] * 0.85, Ls[-1] * 1.35)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("window length (bp)")
        ax.set_title(label, fontsize=10)
        ax.grid(axis="y", color="0.92", linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0][0].set_ylabel("frame called correctly (held-out windows)")

    handles = [
        plt.Line2D([], [], color=c, linestyle=ls, linewidth=2, marker="o", label=n)
        for _, n, c, ls in _FRAME_SERIES
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=8)
    fig.suptitle(
        title or "Which letter of the codon is this? Reading the frame from a landscape",
        fontsize=11,
    )
    fig.subplots_adjust(bottom=0.27, top=0.87, wspace=0.12)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


_CODON_COLORS = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}


def render_gene_landscape(scan, expected_gc, codon_pos, out_path="gene_landscape.png", title=None):
    """A saturation landscape drawn against the annotated reading frame (Part 11).

    Top: the usual heatmap (alt base x position, diverging LLR centered at 0), with
    thin guides at codon boundaries. Bottom, sharing x: the model's expected G+C at
    each site *before seeing it*, one bar per position colored by codon position
    (1/2/3 on the gene's strand; gray where no gene covers it).

    Args:
        scan: GenomeModel.saturation_scan dict.
        expected_gc: per scanned position, the model's P(G)+P(C) (site_profile).
        codon_pos: per scanned position, 1/2/3 or 0 (not in a single CDS).

    Raises:
        ImportError: If matplotlib is not installed (dev-only dependency).
        ValueError: If the scan is empty or the tracks don't line up.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")  # headless: no display needed to write a PNG
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import TwoSlopeNorm
    except ImportError as e:  # pragma: no cover - exercised only without matplotlib
        raise ImportError(
            "render_gene_landscape needs matplotlib. Install it with "
            "`pip install matplotlib` (it is a dev-only, optional dependency)."
        ) from e

    grid = scan["grid"]
    positions = scan["positions"]
    if not grid:
        raise ValueError("scan has no scored positions to plot (n_scored == 0)")
    if not (len(expected_gc) == len(codon_pos) == len(positions)):
        raise ValueError("expected_gc and codon_pos must align with scan positions")

    data = np.asarray(grid, dtype=float).T
    vmax = float(np.abs(data).max()) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    n = len(positions)
    fig_w = max(7.0, min(24.0, n * 0.09))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(fig_w, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1.0]}
    )
    im = ax1.imshow(data, aspect="auto", cmap="RdBu", norm=norm, interpolation="nearest")
    ax1.set_yticks(range(len(ALT_BASES)))
    ax1.set_yticklabels(ALT_BASES)
    ax1.set_ylabel("alt base")
    # codon boundary guides: before each codon position 1 (gene orientation)
    for i in range(1, n):
        if codon_pos[i] and codon_pos[i - 1] and {codon_pos[i], codon_pos[i - 1]} == {1, 3}:
            ax1.axvline(i - 0.5, color="0.25", linewidth=0.4, alpha=0.6)
    cbar = fig.colorbar(im, ax=[ax1, ax2], fraction=0.02, pad=0.01)
    cbar.set_label("LLR (alt − ref), nats")

    colors = [_CODON_COLORS.get(c, "#b0afa9") for c in codon_pos]
    ax2.bar(range(n), expected_gc, width=0.8, color=colors)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("expected G+C\n(before seeing it)", fontsize=8)
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax2.grid(axis="y", color="0.92", linewidth=0.6)
    ax2.set_axisbelow(True)
    step = max(1, n // 20)
    ax2.set_xticks(list(range(0, n, step)))
    ax2.set_xticklabels([str(positions[i]) for i in range(0, n, step)], rotation=90, fontsize=7)
    ax2.set_xlabel("genome position (bp)")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_CODON_COLORS[c], label=f"codon position {c}")
        for c in (1, 2, 3)
    ]
    ax2.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=3,
        frameon=False,
        fontsize=8,
    )
    if title:
        ax1.set_title(title)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
