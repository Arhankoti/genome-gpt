from config import COMPLEMENT, ITOS, SEP_ID, STOI, VOCAB_SIZE


def test_vocab_size():
    assert VOCAB_SIZE == 6


def test_stoi_itos_roundtrip():
    for c, i in STOI.items():
        assert ITOS[i] == c


def test_complement_map():
    assert COMPLEMENT[STOI["A"]] == STOI["T"]
    assert COMPLEMENT[STOI["T"]] == STOI["A"]
    assert COMPLEMENT[STOI["C"]] == STOI["G"]
    assert COMPLEMENT[STOI["G"]] == STOI["C"]
    assert COMPLEMENT[STOI["N"]] == STOI["N"]
    assert COMPLEMENT[STOI["|"]] == STOI["|"]


def test_complement_involution():
    """double-complement is identity for all tokens"""
    for i in range(VOCAB_SIZE):
        assert COMPLEMENT[COMPLEMENT[i]] == i


def test_sep_id():
    assert SEP_ID == STOI["|"]
