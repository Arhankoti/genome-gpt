"""Tests for the whole-genome held-out split in data/prepare.py (Part 5).

All offline: builds a tiny multi-record FASTA in tmp_path. The split correctness
(held-out genomes absent from train, deterministic, disjoint) is the honest-
benchmark foundation, so it is pinned tightly.
"""

import textwrap

import numpy as np
import pytest

from config import SEP_ID
from data.prepare import (
    _join_with_boundaries,
    encode,
    read_named_records,
    select_holdout,
)

FASTA = textwrap.dedent("""\
    >genome_a desc one
    ACGTACGTAC
    >genome_b desc two
    TTTTCCCCGG
    >genome_c
    GGGGAAAATT
    >genome_d
    CACACACACA
""")


@pytest.fixture
def fasta_file(tmp_path):
    p = tmp_path / "multi.fasta"
    p.write_text(FASTA)
    return str(p)


# --- read_named_records ---


def test_named_records_parse_names_and_seqs(fasta_file):
    named = read_named_records(fasta_file)
    assert [n for n, _ in named] == ["genome_a", "genome_b", "genome_c", "genome_d"]
    assert named[0][1] == "ACGTACGTAC"  # first token of header only; sequence uppercased


def test_named_records_name_is_first_header_token(fasta_file):
    named = dict(read_named_records(fasta_file))
    # "genome_a desc one" -> name is "genome_a"
    assert "genome_a" in named


# --- select_holdout ---


def test_named_holdout_selects_exactly_those(fasta_file):
    names = [n for n, _ in read_named_records(fasta_file)]
    chosen = select_holdout(names, holdout_genomes="genome_b,genome_d")
    assert chosen == {"genome_b", "genome_d"}


def test_random_holdout_is_deterministic(fasta_file):
    names = [n for n, _ in read_named_records(fasta_file)]
    a = select_holdout(names, holdout_k=2, holdout_seed=7)
    b = select_holdout(names, holdout_k=2, holdout_seed=7)
    assert a == b
    assert len(a) == 2


def test_random_holdout_seed_changes_pick(fasta_file):
    names = [n for n, _ in read_named_records(fasta_file)]
    # at least one seed pair should differ; guards against ignoring the seed
    picks = {frozenset(select_holdout(names, holdout_k=2, holdout_seed=s)) for s in range(6)}
    assert len(picks) > 1


def test_missing_named_holdout_raises(fasta_file):
    names = [n for n, _ in read_named_records(fasta_file)]
    with pytest.raises(ValueError, match="not found"):
        select_holdout(names, holdout_genomes="genome_z")


def test_holdout_cannot_take_all_records(fasta_file):
    names = [n for n, _ in read_named_records(fasta_file)]
    with pytest.raises(ValueError):
        select_holdout(names, holdout_k=len(names))


def test_holdout_needs_two_records():
    with pytest.raises(ValueError, match="at least 2"):
        select_holdout(["solo"], holdout_k=1)


# --- split correctness (the core guarantee) ---


def _split(named, holdout):
    train_seqs = [s for n, s in named if n not in holdout]
    val_seqs = [s for n, s in named if n in holdout]
    return _join_with_boundaries(train_seqs), _join_with_boundaries(val_seqs)


def test_heldout_genome_absent_from_train(fasta_file):
    named = read_named_records(fasta_file)
    holdout = {"genome_b", "genome_d"}
    train, val = _split(named, holdout)
    # every held-out genome's exact tokens must appear in val and NOT in train
    train_bytes = train.tobytes()
    for name, seq in named:
        seg = encode(seq).tobytes()
        if name in holdout:
            assert seg in val.tobytes()
            assert seg not in train_bytes  # the whole point: no leakage
        else:
            assert seg in train_bytes


def test_boundary_token_count_per_split(fasta_file):
    named = read_named_records(fasta_file)
    holdout = {"genome_b", "genome_d"}  # 2 val, 2 train
    train, val = _split(named, holdout)
    # SEP count == records_in_split - 1
    assert int((train == SEP_ID).sum()) == 1
    assert int((val == SEP_ID).sum()) == 1


def test_splits_are_disjoint_lengths(fasta_file):
    named = read_named_records(fasta_file)
    holdout = {"genome_c"}
    train, val = _split(named, holdout)
    # 3 train records + 2 boundaries; 1 val record + 0 boundaries
    assert int((train != SEP_ID).sum()) == 30  # 3 * 10 bp
    assert int((val != SEP_ID).sum()) == 10
    assert np.uint8(SEP_ID) not in val  # single val record => no internal boundary


def test_join_empty_is_empty():
    assert _join_with_boundaries([]).size == 0
