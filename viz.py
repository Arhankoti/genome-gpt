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
