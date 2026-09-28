"""Tests for the Part 10 variant layer.

The pure helpers in variants.py are pinned by construction (codon table, class
labels, feature-table parsing, AUC, verdict tiers). The model side
(GenomeModel.variant_effects) is checked only for internal consistency against a
hand-rolled forward pass on the untrained tiny checkpoint — LLRs are noise there,
so never assert a sign or a class ordering (same discipline as Parts 2-9).
"""

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from config import STOI, VOCAB_SIZE
from evaluate import _contexts, markov_fit, markov_variant_llr
from inference import GenomeModel
from variants import (
    CODON_TABLE,
    classify_substitution,
    codon_variants,
    disruption_percentile,
    neighborhood_variants,
    parse_feature_table,
    reverse_complement,
    separation_auc,
    variant_report,
    variant_verdict,
)


def _rand_dna(n, seed):
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n))


@pytest.fixture(scope="module")
def gm(tiny_ckpt):
    return GenomeModel(tiny_ckpt)


# --- GenomeModel.variant_effects (internal consistency only) ---


@torch.no_grad()
def _window_ll(gm, s):
    ids = torch.tensor([STOI[c] for c in s])
    logits, _ = gm.model(ids[:-1].unsqueeze(0))
    logp = F.log_softmax(logits[0], dim=-1)
    return float(logp[torch.arange(len(ids) - 1), ids[1:]].sum())


def test_variant_effects_matches_hand_rolled(gm, tiny_cfg):
    seq = _rand_dna(80, 0)
    W = tiny_cfg.block_size
    pos, alt = 40, "A" if seq[40] != "A" else "C"
    lo = pos - W // 2
    ref_w = seq[lo : lo + W]
    alt_w = ref_w[: pos - lo] + alt + ref_w[pos - lo + 1 :]
    expected = _window_ll(gm, alt_w) - _window_ll(gm, ref_w)
    got = gm.variant_effects(seq, [(pos, alt)])[0]
    assert got == pytest.approx(expected, abs=1e-4)


def test_variant_effects_edge_window_clamped(gm, tiny_cfg):
    seq = _rand_dna(80, 1)
    W = tiny_cfg.block_size
    for pos in (0, 3, 79):
        alt = "G" if seq[pos] != "G" else "T"
        lo = max(0, min(pos - W // 2, len(seq) - W))
        ref_w = seq[lo : lo + W]
        alt_w = ref_w[: pos - lo] + alt + ref_w[pos - lo + 1 :]
        expected = _window_ll(gm, alt_w) - _window_ll(gm, ref_w)
        assert gm.variant_effects(seq, [(pos, alt)])[0] == pytest.approx(expected, abs=1e-4)


def test_variant_effects_batch_invariant_and_ordered(gm):
    seq = _rand_dna(100, 2)
    vs = neighborhood_variants(seq, 50, 10)
    a = gm.variant_effects(seq, vs, batch_size=1)
    b = gm.variant_effects(seq, vs, batch_size=64)
    assert len(a) == len(vs)
    assert np.allclose(a, b, atol=1e-4)


def test_variant_effects_same_base_is_zero(gm):
    seq = _rand_dna(60, 3)
    assert gm.variant_effects(seq, [(30, seq[30])]) == [0.0]


def test_variant_effects_short_sequence(gm):
    # shorter than block_size: the window shrinks to the whole sequence
    r = gm.variant_effects("ACGTACGTAC", [(5, "A")])
    assert len(r) == 1 and np.isfinite(r[0])


@pytest.mark.parametrize("pos,alt", [(-1, "A"), (60, "A"), (10, "N"), (10, "X")])
def test_variant_effects_rejects_bad_input(gm, pos, alt):
    with pytest.raises(ValueError):
        gm.variant_effects(_rand_dna(60, 4), [(pos, alt)])


def test_variant_effect_agrees_with_batched(gm):
    seq = _rand_dna(90, 5)
    alt = "T" if seq[45] != "T" else "A"
    single = gm.variant_effect(seq, 45, alt)["llr"]
    assert single == pytest.approx(gm.variant_effects(seq, [(45, alt)])[0], abs=1e-6)


# --- codon table + classification ---


def test_codon_table_shape():
    assert len(CODON_TABLE) == 64
    assert sorted(c for c, aa in CODON_TABLE.items() if aa == "*") == ["TAA", "TAG", "TGA"]
    assert CODON_TABLE["ATG"] == "M" and CODON_TABLE["TGG"] == "W"
    assert len(set(CODON_TABLE.values())) == 21  # 20 amino acids + stop


def test_classify_substitution():
    assert classify_substitution("CTG", 2, "A") == "synonymous"  # Leu -> Leu
    assert classify_substitution("CTG", 0, "A") == "missense"  # Leu -> Met
    assert classify_substitution("TGG", 2, "A") == "nonsense"  # Trp -> stop
    assert classify_substitution("TAA", 2, "C") == "stop_loss"
    assert classify_substitution("TAA", 1, "G") == "stop_to_stop"
    with pytest.raises(ValueError):
        classify_substitution("CTG", 0, "C")


def test_reverse_complement():
    assert reverse_complement("AACGTN") == "NACGTT"


# --- feature table parsing ---

_FT = "\n".join(
    [
        ">Feature ref|NC_TEST|",
        "1\t9\tgene",
        "\t\t\tgene\tabc",
        "1\t9\tCDS",
        "\t\t\tproduct\tplus-strand protein",
        "30\t19\tCDS",
        "\t\t\tproduct\tminus-strand protein",
        "40\t45\tCDS",
        "47\t50",
        "\t\t\tproduct\tjoined (frameshift) protein",
        "<60\t69\tCDS",
        "\t\t\tproduct\tpartial",
        "70\t78\tCDS",
        "\t\t\tpseudo",
        "80\t84\tCDS",
        "\t\t\tproduct\tnot a multiple of 3",
    ]
)


def test_parse_feature_table_keeps_only_clean_cds():
    cds = parse_feature_table(_FT)
    assert cds == [
        {"start": 0, "end": 9, "strand": "+"},
        {"start": 18, "end": 30, "strand": "-"},
    ]


# --- codon_variants ---


def test_codon_variants_plus_strand():
    genome = "ATGCTGTAA"
    rows = codon_variants(genome, {"start": 0, "end": 9, "strand": "+"}, 1)  # CTG
    assert len(rows) == 9
    assert {r["pos"] for r in rows} == {3, 4, 5}
    for r in rows:
        assert r["ref"] == genome[r["pos"]] and r["alt"] != r["ref"]
    syn = [r for r in rows if r["klass"] == "synonymous"]
    assert {(r["pos"], r["alt"]) for r in syn} >= {(5, "A"), (5, "C"), (5, "T"), (3, "T")}


def test_codon_variants_minus_strand_matches_rc():
    gene = "ATGTGGCTGTAA"  # on the minus strand: + strand holds its reverse complement
    genome = "GG" + reverse_complement(gene) + "CC"
    cds = {"start": 2, "end": 2 + len(gene), "strand": "-"}
    rows = codon_variants(genome, cds, 1)  # TGG (Trp)
    assert len(rows) == 9
    for r in rows:
        assert genome[r["pos"]] == r["ref"]
        # apply the + strand change, read the gene back, and re-classify independently
        mutated = genome[: r["pos"]] + r["alt"] + genome[r["pos"] + 1 :]
        new_gene = reverse_complement(mutated[cds["start"] : cds["end"]])
        new_aa = CODON_TABLE[new_gene[3:6]]
        expect = "nonsense" if new_aa == "*" else "synonymous" if new_aa == "W" else "missense"
        assert r["klass"] == expect
    assert sum(r["klass"] == "nonsense" for r in rows) == 2  # TGG -> TAG, TGA


def test_codon_variants_gc_neutral_flag():
    rows = codon_variants("ATGCTGTAA", {"start": 0, "end": 9, "strand": "+"}, 1)
    for r in rows:
        assert r["gc_neutral"] == ((r["ref"] in "GC") == (r["alt"] in "GC"))


# --- separation_auc / percentile ---


def test_separation_auc_extremes_and_ties():
    assert separation_auc([-3, -2], [0, 1]) == 1.0
    assert separation_auc([0, 1], [-3, -2]) == 0.0
    assert separation_auc([0, 0], [0, 0]) == 0.5
    assert np.isnan(separation_auc([], [1.0]))


def test_separation_auc_matches_brute_force():
    rng = np.random.default_rng(0)
    d, b = rng.normal(-1, 1, 40).round(1), rng.normal(0, 1, 50).round(1)
    brute = np.mean([(x < y) + 0.5 * (x == y) for x in d for y in b])
    assert separation_auc(d, b) == pytest.approx(brute)


def test_disruption_percentile():
    bg = [-1.0, -0.5, 0.0, 0.5]
    assert disruption_percentile(-2.0, bg) == 1.0
    assert disruption_percentile(1.0, bg) == 0.0
    assert disruption_percentile(-0.5, bg) == pytest.approx(0.625)  # 2 above + half a tie


# --- variant_verdict tiers (by construction) ---

_BG = list(np.linspace(-2.0, 0.0, 100))


def test_verdict_likely_disruptive():
    v = variant_verdict(-5.0, _BG)
    assert v["verdict"] == "likely_disruptive"
    assert v["disruption_percentile"] == 1.0
    assert v["robust_z"] < 0


def test_verdict_uncertain_and_tolerated():
    assert variant_verdict(-1.5, _BG)["verdict"] == "uncertain"
    assert variant_verdict(-0.2, _BG)["verdict"] == "likely_tolerated"
    # a positive LLR is tolerated even if the neighbors are all positive too
    assert variant_verdict(0.1, [0.5] * 100)["verdict"] == "likely_tolerated"


def test_verdict_unreliable_context_overrides():
    for ctx in ("random_like", "low_complexity"):
        v = variant_verdict(-5.0, _BG, context_verdict=ctx)
        assert v["verdict"] == "unreliable_context"
        assert v["disruption_percentile"] == 1.0  # the numbers still come back


def test_verdict_cautions_and_confidence():
    v = variant_verdict(-5.0, _BG[:20], left_context=5)
    assert v["confidence"] == "low"
    assert any("background" in c for c in v["cautions"])
    assert any("left context" in c for c in v["cautions"])
    assert variant_verdict(-5.0, _BG)["confidence"] == "moderate"


# --- variant_report composition with fake closures ---


def test_variant_report_composes_target_and_background():
    seq = _rand_dna(300, 6)
    calls = []

    def effects_fn(s, vs):
        calls.append(vs)
        return [-9.0] + [-0.1] * (len(vs) - 1)

    alt = "A" if seq[150] != "A" else "G"
    v = variant_report(seq, 150, alt, effects_fn=effects_fn, score_fn=lambda s: 1.8, flank=10)
    assert calls[0][0] == (150, alt)  # target scored first, in the same batch
    assert (150, alt) not in calls[0][1:]
    assert v["n_background"] == 3 * 21 - 1
    assert v["ref_base"] == seq[150] and v["alt_base"] == alt
    # score_fn gives shuffle == real, so the context reads as composition-only (not
    # random/low-complexity) and the verdict stands
    assert v["verdict"] == "likely_disruptive"


def test_variant_report_rejects_bad_input():
    seq = _rand_dna(100, 7)
    kw = {"effects_fn": lambda s, vs: [0.0] * len(vs), "score_fn": lambda s: 1.9}
    with pytest.raises(ValueError):
        variant_report(seq, 50, seq[50], **kw)
    with pytest.raises(ValueError):
        variant_report(seq, 500, "A", **kw)
    with pytest.raises(ValueError):
        variant_report(seq, 50, "N", **kw)


def test_variant_report_random_context_is_unreliable():
    seq = _rand_dna(300, 8)
    alt = "A" if seq[150] != "A" else "C"
    v = variant_report(
        seq,
        150,
        alt,
        effects_fn=lambda s, vs: [-9.0] + [-0.1] * (len(vs) - 1),
        score_fn=lambda s: 2.0,  # at the random line
    )
    assert v["verdict"] == "unreliable_context"


# --- Markov baseline for variants ---


def test_markov_variant_llr_matches_full_rescore():
    rng = np.random.default_rng(9)
    train = rng.integers(0, 4, 5000).astype(np.uint8)
    k = 3
    probs = markov_fit(train, k)
    ids = rng.integers(0, 4, 60).astype(np.int64)

    def full_ll(s):
        ctx, nxt = _contexts(s, k)
        return np.log(probs[ctx, nxt]).sum()

    for pos in (0, 2, 30, 59):
        alt_id = (int(ids[pos]) + 1) % 4
        alt = ids.copy()
        alt[pos] = alt_id
        expected = full_ll(alt) - full_ll(ids)
        assert markov_variant_llr(probs, k, ids, pos, alt_id) == pytest.approx(expected)
    assert probs.shape == (VOCAB_SIZE**k, VOCAB_SIZE)


# --- codon-test summary + figure ---


def _fake_rows():
    rng = np.random.default_rng(10)
    rows = []
    for klass, mu in (("synonymous", 0.0), ("missense", -0.5), ("nonsense", -2.0)):
        for i in range(12):
            rows.append(
                {
                    "klass": klass,
                    "codon_pos": 1 + i % 3,
                    "gc_neutral": i % 2 == 0,
                    "ref": "GCAT"[i % 4],
                    "alt": "ATGC"[i % 4],
                    "llr_neural": float(rng.normal(mu, 0.3)),
                    "llr_markov": float(rng.normal(mu / 2, 0.3)),
                    "verdict": "likely_disruptive" if klass == "nonsense" else "uncertain",
                }
            )
    return rows


def test_summarize_classes_and_auc():
    from detective import summarize

    s = summarize(_fake_rows())
    assert s["n_variants"] == 36
    assert s["classes"]["nonsense"]["n"] == 12
    assert s["classes"]["nonsense"]["frac_likely_disruptive"] == 1.0
    assert s["classes"]["synonymous"]["frac_likely_disruptive"] == 0.0
    # by construction nonsense sits far below synonymous
    assert s["auc"]["all"]["nonsense_vs_synonymous"]["llr_neural"] > 0.9
    assert s["auc"]["gc_neutral"]["nonsense_vs_synonymous"]["n"] == [6, 6]
    assert set(s["codon_position"]) == {"1", "2", "3"}
    assert 0.0 <= s["classes"]["missense"]["sign_negative_frac"] <= 1.0
    # ref G/C -> alt A/T for i%4 in (0, 1); A/T -> G/C for (2, 3)
    shift = s["gc_shift"]
    assert sum(shift[cp]["S>W"]["n"] + shift[cp]["W>S"]["n"] for cp in shift) == 36
    assert set(shift["1"]["S>W"]) == {"n", "llr_neural", "llr_markov"}


def test_render_codon_style(tmp_path):
    pytest.importorskip("matplotlib")
    from detective import summarize
    from viz import render_codon_style

    res = {"rows": _fake_rows()}
    res["summary"] = summarize(res["rows"])
    out = render_codon_style([res, res], ["a", "b"], out_path=str(tmp_path / "c.png"))
    assert (tmp_path / "c.png").stat().st_size > 0 and out.endswith("c.png")
    with pytest.raises(ValueError):
        render_codon_style([res], ["a", "b"], out_path=str(tmp_path / "x.png"))


def test_sample_codons_respects_margin_and_is_seeded():
    from detective import sample_codons

    genome = _rand_dna(5000, 11)
    cds = [{"start": 100, "end": 4900, "strand": "+"}, {"start": 200, "end": 4700, "strand": "-"}]
    a = sample_codons(genome, cds, 30, margin=300, seed=1)
    assert a == sample_codons(genome, cds, 30, margin=300, seed=1)
    assert len(a) == 30 and len({(c["start"], i) for c, i in a}) == 30
    for c, i in a:
        g0 = c["start"] + 3 * i if c["strand"] == "+" else c["end"] - 3 * i - 3
        assert 300 <= g0 and g0 + 303 <= len(genome)


def test_render_variant_classes(tmp_path):
    pytest.importorskip("matplotlib")
    from detective import summarize
    from viz import render_variant_classes

    rows = _fake_rows()
    out = render_variant_classes(
        {"rows": rows, "summary": summarize(rows), "meta": {"genome": "test"}},
        out_path=str(tmp_path / "v.png"),
    )
    assert (tmp_path / "v.png").stat().st_size > 0 and out.endswith("v.png")
    with pytest.raises(ValueError):
        render_variant_classes({"rows": [], "summary": {}}, out_path=str(tmp_path / "x.png"))
