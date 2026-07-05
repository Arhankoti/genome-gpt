"""Tests for bridge/tools.py and bridge/schemas.py."""

from unittest.mock import MagicMock, patch

# --- schemas ---


def test_anthropic_tools_structure():
    from bridge.schemas import ANTHROPIC_TOOLS

    assert len(ANTHROPIC_TOOLS) == 4
    names = {t["name"] for t in ANTHROPIC_TOOLS}
    assert names == {"dna_score", "dna_generate", "dna_variant_effect", "dna_embed"}


def test_anthropic_tools_have_input_schema():
    from bridge.schemas import ANTHROPIC_TOOLS

    for tool in ANTHROPIC_TOOLS:
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"


def test_openai_functions_derived_from_anthropic():
    from bridge.schemas import ANTHROPIC_TOOLS, OPENAI_FUNCTIONS

    assert len(OPENAI_FUNCTIONS) == len(ANTHROPIC_TOOLS)
    for oai, ant in zip(OPENAI_FUNCTIONS, ANTHROPIC_TOOLS):
        assert oai["type"] == "function"
        assert oai["function"]["name"] == ant["name"]


# --- dispatch ---


def _make_mock_model():
    m = MagicMock()
    m.score.return_value = {
        "mean_logprob": -1.4,
        "perplexity": 4.0,
        "bits_per_bp": 2.0,
        "n_scored": 8,
    }
    m.generate.return_value = "ACGTACGT"
    m.variant_effect.return_value = {
        "llr": -0.5,
        "ref_base": "A",
        "alt_base": "T",
        "position": 0,
        "interpretation": "more disruptive",
    }
    m.embed.return_value = [0.1] * 16
    return m


def test_dispatch_score():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        result = dispatch("dna_score", {"sequence": "ACGTACGT"})
    assert "bits_per_bp" in result
    mock.score.assert_called_once_with("ACGTACGT")


def test_dispatch_generate():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        result = dispatch("dna_generate", {"prompt": "ACG", "n_bases": 10})
    assert "sequence" in result


def test_dispatch_variant_effect():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        result = dispatch("dna_variant_effect", {"ref_seq": "ACGT" * 4, "pos": 2, "alt_base": "T"})
    assert "llr" in result


def test_dispatch_embed():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        result = dispatch("dna_embed", {"sequence": "ACGT"})
    assert "embedding" in result
    assert "dim" in result


def test_dispatch_unknown_tool():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        result = dispatch("dna_nonexistent", {})
    assert "error" in result


def test_dispatch_generate_temperature_default():
    mock = _make_mock_model()
    with patch("bridge.tools.get_model", return_value=mock):
        from bridge.tools import dispatch

        dispatch("dna_generate", {"prompt": "A", "n_bases": 5})
    _, kwargs = mock.generate.call_args
    assert kwargs.get("temperature", 0.8) == 0.8
