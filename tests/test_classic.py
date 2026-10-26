"""Tests for the Part 12 classic tools.

classic.py is pure and deterministic, so unlike the neural tools every answer can
be pinned exactly on hand-built sequences: ORFs on both strands with known
coordinates, G+C / skew by counting, restriction sites and digests by hand.
"""

import numpy as np
import pytest

from classic import (
    ENZYMES,
    digest,
    find_orfs,
    gc_fraction,
    gc_report,
    gc_scan,
    gc_skew_extrema,
    gene_location,
    orf_report,
    orf_shuffle_null,
    orf_stop_key,
    restriction_sites,
    reverse_complement,
    shuffle_p,
    site_enrichment,
    sites_report,
)

GENE = "ATG" + "GCT" * 5 + "TAA"  # 5 sense codons after the start, then a stop


def _rand_dna(n, seed, gc=0.5):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n, p=p))


# --- ORFs ---


def test_find_orfs_plus_strand_coords():
    seq = "CC" + GENE + "CC"
    orfs = find_orfs(seq, min_codons=1)
    plus = [o for o in orfs if o["strand"] == "+"]
    assert plus == [
        {"start": 2, "end": 23, "strand": "+", "frame": 2, "n_codons": 6, "start_codon": "ATG"}
    ]
    assert seq[plus[0]["start"] : plus[0]["end"]] == GENE


def test_find_orfs_minus_strand_coords():
    seq = "CC" + reverse_complement(GENE) + "CCC"
    minus = [o for o in find_orfs(seq, min_codons=1) if o["strand"] == "-"]
    assert len(minus) == 1
    o = minus[0]
    assert reverse_complement(seq[o["start"] : o["end"]]) == GENE
    assert o["n_codons"] == 6


def test_find_orfs_longest_form_and_alt_starts():
    # an upstream GTG then an in-frame ATG: one ORF per stop, from the earliest start
    seq = "GTG" + "GCT" + GENE
    o = [x for x in find_orfs(seq, min_codons=1) if x["strand"] == "+"]
    assert len(o) == 1 and o[0]["start"] == 0 and o[0]["start_codon"] == "GTG"
    assert find_orfs(seq, min_codons=1, starts=("ATG",))[0]["start"] == 6


def test_find_orfs_min_codons_and_incomplete():
    assert [o for o in find_orfs(GENE, min_codons=7) if o["strand"] == "+"] == []
    # no stop before the end: not reported
    assert [o for o in find_orfs("ATG" + "GCT" * 10, min_codons=1) if o["strand"] == "+"] == []
    # an N inside breaks the ORF
    assert [o for o in find_orfs("ATGGCNGCTTAA", min_codons=1) if o["strand"] == "+"] == []


def test_find_orfs_matches_bruteforce():
    seq = _rand_dna(3000, 1, gc=0.6)
    stops, starts = {"TAA", "TAG", "TGA"}, {"ATG", "GTG", "TTG"}
    want = set()
    for strand, s in (("+", seq), ("-", reverse_complement(seq))):
        for f in range(3):
            open_at = None
            for i in range(f, len(s) - 2, 3):
                c = s[i : i + 3]
                if c in stops:
                    if open_at is not None and (i - open_at) // 3 >= 20:
                        lo, hi = open_at, i + 3
                        want.add(
                            (lo, hi, "+") if strand == "+" else (len(s) - hi, len(s) - lo, "-")
                        )
                    open_at = None
                elif c in starts and open_at is None:
                    open_at = i
    got = {(o["start"], o["end"], o["strand"]) for o in find_orfs(seq, min_codons=20)}
    assert got == want


def test_gc_rich_dna_has_longer_chance_orfs():
    # the Part 12 headline in miniature: stops are A/T-rich
    rich = orf_shuffle_null(_rand_dna(20000, 2, gc=0.7), n_shuffles=3)
    poor = orf_shuffle_null(_rand_dna(20000, 3, gc=0.35), n_shuffles=3)
    assert rich.min() > poor.max()


def test_orf_shuffle_null_deterministic_and_p():
    seq = _rand_dna(2000, 4)
    a, b = orf_shuffle_null(seq, 5, seed=1), orf_shuffle_null(seq, 5, seed=1)
    assert np.array_equal(a, b) and a.shape == (5,)
    assert shuffle_p(10**6, a) == pytest.approx(1 / 6)
    assert shuffle_p(0, a) == 1.0


def test_orf_report_bounded():
    seq = _rand_dna(5000, 5, gc=0.65)
    r = orf_report(seq, min_codons=30, top_k=3, n_shuffles=4)
    assert len(r["longest_orfs"]) <= 3 and r["n_shuffles"] == 4
    assert r["n_orfs"] == r["n_plus"] + r["n_minus"]
    lens = [o["n_codons"] for o in r["longest_orfs"]]
    assert lens == sorted(lens, reverse=True)
    assert all(0 < o["shuffle_p"] <= 1 for o in r["longest_orfs"])
    assert seq not in str(r)


def test_orf_stop_key_matches_cds_convention():
    assert orf_stop_key({"start": 10, "end": 40, "strand": "+"}) == ("+", 40)
    assert orf_stop_key({"start": 10, "end": 40, "strand": "-"}) == ("-", 10)


# --- G+C / skew ---


def test_gc_fraction_and_scan():
    assert gc_fraction("GGCCAATT") == 0.5
    assert gc_fraction("GCNN") == 1.0
    assert gc_fraction("NNN") == 0.0
    s = gc_scan("GGGG" + "AAAA" + "CCCC", window=4)
    assert s["starts"] == [0, 4, 8]
    assert s["gc"] == [1.0, 0.0, 1.0]
    assert s["skew"][0] == 1.0 and s["skew"][2] == -1.0 and np.isnan(s["skew"][1])
    assert s["cumulative_skew"] == [0, 4, 4]
    # shorter than the window: one window over everything
    assert gc_scan("GCA", window=100)["gc"] == [pytest.approx(2 / 3)]


def test_gc_skew_extrema_finds_origin():
    # cumulative G-C climbs through a G-rich block and bottoms out after a longer C block
    seq = _rand_dna(200, 6) + "G" * 60 + _rand_dna(200, 7) + "C" * 120 + _rand_dna(200, 8)
    e = gc_skew_extrema(seq)
    assert 255 <= e["terminus_guess"] <= 470
    assert e["origin_guess"] >= 570
    assert e["skew_range"] >= 100
    assert 0 <= e["separation"] <= 0.5
    with pytest.raises(ValueError):
        gc_skew_extrema("")


def test_gc_report_bounded():
    r = gc_report(_rand_dna(50000, 9), window=100, max_points=40)
    assert r["n_windows"] == 500 and len(r["profile"]) <= 40
    assert r["gc_min"]["gc"] <= r["gc"] <= r["gc_max"]["gc"]


# --- restriction sites ---


def test_restriction_sites_palindrome_and_iupac():
    seq = "AAGAATTCAAGAATTC"
    hits = restriction_sites(seq, ["EcoRI"])["EcoRI"]
    assert [h["pos"] for h in hits] == [2, 10]
    assert [h["cut"] for h in hits] == [3, 11]  # G^AATTC
    assert all(h["strand"] == "+" for h in hits)
    assert [h["pos"] for h in restriction_sites("GACTCGAATC", ["HinfI"])["HinfI"]] == [0, 5]


def test_restriction_sites_non_palindromic_both_strands():
    seq = "AAGGTCTCAAAA" + reverse_complement("GGTCTC") + "AAA"
    hits = restriction_sites(seq, ["BsaI"])["BsaI"]
    assert [(h["pos"], h["strand"]) for h in hits] == [(2, "+"), (12, "-")]
    assert hits[0]["cut"] == 2 + 7


def test_restriction_unknown_enzyme():
    with pytest.raises(ValueError):
        restriction_sites("ACGT", ["NoSuchI"])


def test_digest_linear_and_circular():
    assert digest(100, [30, 70]) == [30, 40, 30]
    assert digest(100, [30, 70], circular=True) == [40, 60]
    assert digest(100, []) == [100]
    assert digest(100, [0, 100, 150, 50, 50]) == [50, 50]  # out of range / duplicate cuts


def test_sites_report_bounded():
    seq = ("GAATTC" + "A" * 20) * 50
    r = sites_report(seq, ["EcoRI", "NotI"], max_positions=5)
    assert r["enzymes"]["EcoRI"]["n_sites"] == 50
    assert len(r["enzymes"]["EcoRI"]["cut_positions"]) == 5
    assert sum(digest(len(seq), [3 + 26 * i for i in range(50)])) == len(seq)
    assert r["cutters"] == ["EcoRI"] and r["non_cutters"] == ["NotI"]


def test_site_enrichment_shapes():
    e = site_enrichment(_rand_dna(5000, 10), ["EcoRI", "HaeIII"], n_shuffles=2)
    assert set(e) == {"EcoRI", "HaeIII"}
    assert e["HaeIII"]["observed"] > 0 and e["HaeIII"]["ratio"] > 0
    assert set(ENZYMES) >= set(e)


def test_gene_location_from_feature_table():
    ft = (
        ">Feature ref|X|\n1\t30\tgene\n\t\t\tgene\tfoo\n1\t30\tCDS\n\t\t\tproduct\tFoo\n"
        "90\t61\tgene\n\t\t\tgene\tdnaA\n"
    )
    assert gene_location(ft, "foo") == {"start": 0, "end": 30, "strand": "+"}
    assert gene_location(ft, "dnaA") == {"start": 60, "end": 90, "strand": "-"}
    assert gene_location(ft, "bar") is None
