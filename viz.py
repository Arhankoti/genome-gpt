"""Render a saturation-mutagenesis scan as a heatmap PNG.

Matplotlib is an optional, dev-only dependency: it is imported lazily inside the
function so importing this module (or the rest of the package) never requires it.
The only input is the dict returned by GenomeModel.saturation_scan — this module
has no coupling to the model itself.
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
