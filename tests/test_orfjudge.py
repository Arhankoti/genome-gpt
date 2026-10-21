"""Tests for the Part 13 gene-vs-shadow layer.

orfjudge.py is PURE, so its pieces are pinned exactly on hand-built inputs. The
model side (GenomeModel.base_probs) is checked only for internal consistency on the
untrained tiny checkpoint — against site_profile and for the causal-prefix property
— never for whether the untrained model can tell a gene from a shadow (same
discipline as Parts 2-12).
"""

import numpy as np
import pytest

from classic import find_orfs, reverse_complement
from evaluate import markov_base_probs, markov_fit, markov_site_profile
from inference import GenomeModel
from orfjudge import (
    antisense_pairs,
    auc_higher,
    centered_cosine,
    codon_usage,
    gc_frame_rule,
    judge_pairs,
    onehot,
    oriented,
    pair_accuracy,
    phase_profile,
    self_trained_scores,
    trained_scores,
    zscore,
)
from variants import separation_auc


def _rand_dna(n, seed, gc=0.5):
    p = [(1 - gc) / 2, gc / 2, gc / 2, (1 - gc) / 2]
    return "".join(np.random.default_rng(seed).choice(list("ACGT"), n, p=p))


def _uniform(s):
    return np.full((len(s), 4), 0.25)


# --- pieces ---


def test_oriented_both_strands():
    seq = "AAAA" + "ATGCCCTAA" + "GGGG"
    s, off = oriented(seq, {"start": 4, "end": 13, "strand": "+"}, ctx=2)
    assert (s, off) == ("AAATGCCCTAA", 2)
    rc = reverse_complement("ATGCCCTAA")
    seq = "AAAA" + rc + "GGGG"
    s, off = oriented(seq, {"start": 4, "end": 13, "strand": "-"}, ctx=3)
    assert s[off:] == "ATGCCCTAA" and s[:off] == reverse_complement("GGG")
    # context clipped at the sequence end
    assert oriented(seq, {"start": 4, "end": 13, "strand": "-"}, ctx=50)[1] == 4


def test_onehot_and_phase_profile():
    oh = onehot("ACGN")
    assert oh[:3].tolist() == np.eye(4)[:3].tolist() and np.isnan(oh[3]).all()
    prof = phase_profile(onehot("GCA" * 10))
    assert prof.tolist() == [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0]
    # trim skips leading rows; NaN rows are ignored
    p = np.vstack([np.full((3, 4), np.nan), onehot("GCA" * 3)])
    assert phase_profile(p, trim=3).tolist() == prof.tolist()
    assert gc_frame_rule(prof) == pytest.approx(0 - 1)  # GC3 = 0, GC1 = GC2 = 1


def test_codon_usage():
    v = codon_usage("ATGATGCCN")
    assert v.sum() == pytest.approx(1.0)
    assert v[16 * 0 + 4 * 3 + 2] == pytest.approx(1.0)  # ATG twice; CCN skipped
    assert codon_usage("").sum() == 0


def test_centered_cosine_and_zscore():
    assert centered_cosine([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert centered_cosine([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert centered_cosine([1, 1, 1], [1, 2, 3]) == 0.0
    z = zscore([1.0, 2.0, 3.0])
    assert z.mean() == pytest.approx(0) and z.std() == pytest.approx(1)


def test_trained_scores_separate_clusters():
    rng = np.random.default_rng(0)
    a, b = np.array([1, 0, 0, 0, 1, 0]), np.array([0, 1, 0, 1, 0, 0])
    X = np.vstack([a + rng.normal(0, 0.1, (40, 6)), b + rng.normal(0, 0.1, (40, 6))])
    y = np.array([1] * 40 + [0] * 40)
    sc = trained_scores(X, y)
    assert auc_higher(sc[y == 1], sc[y == 0]) > 0.99
    with pytest.raises(ValueError):
        trained_scores(X[:2], np.array([1, 1]))


def test_self_trained_resolves_pairs_toward_the_majority():
    rng = np.random.default_rng(1)
    gene, shadow = np.array([3, 0, 1, 0, 2, 0.0]), np.array([0, 2, 0, 3, 0, 1.0])
    X = np.vstack([gene + rng.normal(0, 0.2, (30, 6)), shadow + rng.normal(0, 0.2, (10, 6))])
    pairs = [(i, 30 + i) for i in range(10)]
    sc = self_trained_scores(X, pairs)
    assert all(sc[i] > sc[j] for i, j in pairs)


def test_antisense_pairs_matches_bruteforce():
    orfs = find_orfs(_rand_dna(20000, 2, gc=0.65), min_codons=40)
    got = antisense_pairs(orfs, min_overlap=0.5)
    want = []
    for i in range(len(orfs)):
        for j in range(i + 1, len(orfs)):
            a, b = orfs[i], orfs[j]
            ov = min(a["end"], b["end"]) - max(a["start"], b["start"])
            short = min(a["end"] - a["start"], b["end"] - b["start"])
            if a["strand"] != b["strand"] and ov >= 0.5 * short:
                want.append((i, j))
    assert got == want and got


def test_auc_and_pair_accuracy():
    assert auc_higher([3, 4], [1, 2]) == 1.0
    assert auc_higher([1, 2], [3, 4]) == 0.0
    assert auc_higher([1, 1], [1, 1]) == 0.5
    pos, neg = np.random.default_rng(3).normal(size=(2, 50))
    assert auc_higher(pos, neg) == pytest.approx(separation_auc(neg, pos))
    labels = [1, 0, 0, 1, 1, 1]
    scores = [2.0, 1.0, 5.0, 3.0, 1.0, 1.0]
    # (0,1): gene wins; (2,3): shadow wins; (4,5): two genes -> skipped
    assert pair_accuracy(scores, [(0, 1), (2, 3), (4, 5)], labels) == 0.5
    assert np.isnan(pair_accuracy(scores, [(4, 5)], labels))


# --- the bridge composition ---


def test_judge_pairs_too_few_orfs():
    r = judge_pairs(_rand_dna(600, 4), _uniform, min_codons=100)
    assert r["verdict"] == "too_few_orfs" and "plain_english" in r


def test_judge_pairs_bounded_and_consistent():
    seq = _rand_dna(60000, 5, gc=0.7)
    r = judge_pairs(seq, _uniform, min_codons=60, top_k=5, max_pool=200)
    assert r["verdict"] == "judged" and r["n_pool"] == 200
    assert len(r["pairs"]) <= 5
    overlaps = [p["overlap_bp"] for p in r["pairs"]]
    assert overlaps == sorted(overlaps, reverse=True)
    for p in r["pairs"]:
        assert p["agree"] == (p["classic_pick"] == p["neural_pick"])
        assert p["plus_orf"]["start"] < p["minus_orf"]["end"]
    assert 0 <= r["agreement_rate"] <= 1
    assert seq[:200] not in str(r)


# --- the model / k-gram anticipation ---


def test_base_probs_matches_site_profile(tiny_ckpt):
    gm = GenomeModel(tiny_ckpt)
    seq = _rand_dna(100, 6)  # > block_size: exercises the sliding window
    p = gm.base_probs(seq)
    assert p.shape == (100, 4) and np.isnan(p[0]).all()
    assert np.allclose(p[1:].sum(axis=1), 1.0)
    egc = np.array(gm.site_profile(seq)["expected_gc"])
    assert np.allclose(p[1:, 1] + p[1:, 2], egc, atol=1e-5)
    assert gm.base_probs("A").shape == (1, 4)


def test_base_probs_is_causal(tiny_ckpt):
    gm = GenomeModel(tiny_ckpt)
    seq = _rand_dna(30, 7)  # within one block: a prefix sees exactly the same context
    full, pre = gm.base_probs(seq), gm.base_probs(seq[:20])
    assert np.allclose(full[1:20], pre[1:], atol=1e-5)


def test_markov_base_probs_matches_site_profile():
    ids = np.random.default_rng(8).integers(0, 4, 3000)
    probs = markov_fit(ids.astype(np.uint8), 3)
    p = markov_base_probs(probs, 3, ids[:200])
    assert np.isnan(p[:3]).all() and np.allclose(p[3:].sum(axis=1), 1.0)
    prof = markov_site_profile(probs, 3, ids[:200])
    assert np.allclose(p[3:, 1] + p[3:, 2], prof["expected_gc"][3:])


def test_shadow_test_end_to_end():
    from together import label_pool, shadow_test

    seq = _rand_dna(30000, 9, gc=0.68)
    orfs = find_orfs(seq, min_codons=60)
    cds = [
        {"start": o["start"], "end": o["end"], "strand": o["strand"]}
        for o in orfs
        if o["strand"] == "+"
    ]  # pretend every + ORF is a gene
    labels = label_pool(seq, cds, orfs)
    assert (labels[[o["strand"] == "+" for o in orfs]] == 1).all()
    assert (labels == 0).any()

    def kgram(s):
        return _uniform(s)

    res = shadow_test(seq, cds, _uniform, kgram, min_codons=60, lengths=(60, 120))
    assert res["n_genes"] + res["n_shadows"] + res["n_other"] == res["n_pool"]
    for name in ("codon_usage", "neural_phase", "together", "gc_frame_rule"):
        for L in ("60", "120"):
            row = res["signals"][name][L]
            assert 0 <= row["self_auc"] <= 1
            if name != "gc_frame_rule":
                assert 0 <= row["trained_auc"] <= 1
    assert 0 <= res["agreement"]["120"]["agree_rate"] <= 1
