"""Tests for data/prepare.py — FASTA parsing and encoding logic."""
import textwrap

import numpy as np
import pytest

from config import STOI
from data.prepare import encode, read_records

SIMPLE_FASTA = textwrap.dedent("""\
    >genome_1
    ACGTACGT
    NNNNACGT
    >genome_2
    TTTTCCCC
    GGGGAAAA
""")


@pytest.fixture
def fasta_file(tmp_path):
    p = tmp_path / "test.fasta"
    p.write_text(SIMPLE_FASTA)
    return str(p)


def test_read_records_count(fasta_file):
    recs = read_records(fasta_file)
    assert len(recs) == 2


def test_read_records_content(fasta_file):
    recs = read_records(fasta_file)
    assert recs[0] == "ACGTACGTNNNNACGT"
    assert recs[1] == "TTTTCCCCGGGGAAAA"


def test_read_records_uppercase(tmp_path):
    p = tmp_path / "lc.fasta"
    p.write_text(">g\nacgt\n")
    recs = read_records(str(p))
    assert recs[0] == "ACGT"


def test_read_records_empty_file(tmp_path):
    p = tmp_path / "empty.fasta"
    p.write_text("")
    assert read_records(str(p)) == []


def test_encode_acgtn():
    arr = encode("ACGTN")
    assert list(arr) == [STOI["A"], STOI["C"], STOI["G"], STOI["T"], STOI["N"]]


def test_encode_sep():
    arr = encode("|")
    assert list(arr) == [STOI["|"]]


def test_encode_unknown_falls_back_to_n():
    arr = encode("XYZ")
    assert all(v == STOI["N"] for v in arr)


def test_encode_dtype():
    arr = encode("ACGT")
    assert arr.dtype == np.uint8
