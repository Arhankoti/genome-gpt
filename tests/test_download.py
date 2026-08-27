"""Tests for the pure (network-free) helpers in data/download.py.

fetch() needs NCBI and is never exercised in CI. relabel_header / seq_sha1 /
manifest_path are pure string functions and are pinned here — relabel_header in
particular is load-bearing: the whole pipeline keys a genome by the friendly name
in its FASTA header (prepare.py --holdout, the leakage guard), so a raw NCBI
accession header would silently break name-based holdout.
"""

from data.download import manifest_path, relabel_header, seq_sha1

RAW = ">NC_000913.3 Escherichia coli str. K-12 substr. MG1655, complete genome\nACGTACGT\nTTGGCCAA"


def test_relabel_header_uses_friendly_name_first_token():
    out = relabel_header(RAW, "e_coli_k12", "NC_000913.3")
    first = out.splitlines()[0]
    assert first == ">e_coli_k12 NC_000913.3"
    # first whitespace token is exactly the name prepare.read_named_records will key on
    assert first[1:].split()[0] == "e_coli_k12"


def test_relabel_header_preserves_sequence_body():
    out = relabel_header(RAW, "e_coli_k12", "NC_000913.3")
    body = "".join(out.splitlines()[1:])
    assert body == "ACGTACGTTTGGCCAA"


def test_relabel_header_seq_content_unchanged():
    # relabeling must not touch the sequence hash — reproducibility id is stable
    assert seq_sha1(RAW) == seq_sha1(relabel_header(RAW, "e_coli_k12", "NC_000913.3"))


def test_relabel_header_handles_headerless_input():
    # degenerate input (no newline) must not raise
    assert relabel_header(">X", "n", "acc") == ">n acc"


def test_seq_sha1_strips_headers_and_uppercases():
    h1, n1 = seq_sha1(">a\nacgt\nACGT")
    h2, n2 = seq_sha1(">different header\nACGTACGT")
    assert n1 == n2 == 8
    assert h1 == h2  # case-insensitive, header-independent content id


def test_manifest_path_is_sibling_json():
    assert manifest_path("data/genomes.fasta") == "data/genomes.manifest.json"
