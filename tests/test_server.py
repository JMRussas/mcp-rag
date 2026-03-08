#  mcp-rag - Server Tests
#
#  Tests for result formatting, LIKE escaping, search filtering,
#  confidence tiers, and gotcha display.

from server import _confidence_tier, _escape_like, _get_gotcha, _sanitize_fts, format_results

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_row_dict(**kwargs):
    """Create a dict that behaves like sqlite3.Row for format_results."""
    defaults = {
        "id": "test:1",
        "text": "sample text",
        "source": "test",
        "module_path": "some.module",
        "type_name": "SomeType",
        "category": "core",
        "heading": "",
        "file_path": "test.py",
    }
    defaults.update(kwargs)
    return defaults


class DictRow(dict):
    """Dict subclass that supports [] access like sqlite3.Row."""

    def __getitem__(self, key):
        return dict.__getitem__(self, key)


def _row(**kwargs) -> DictRow:
    return DictRow(_make_row_dict(**kwargs))


# ---------------------------------------------------------------------------
# format_results tests
# ---------------------------------------------------------------------------


def test_format_results_empty():
    """Empty row list returns 'No results found.'"""
    assert format_results([]) == "No results found."


def test_format_results_single():
    """Single result has header with metadata and text."""
    row = _row(source="engine", module_path="NoZ.Core", type_name="Graphics", file_path="Graphics.cs")
    result = format_results([row])
    assert "Source: engine" in result
    assert "Module: NoZ.Core" in result
    assert "Type: Graphics" in result
    assert "File: Graphics.cs" in result
    assert "sample text" in result


def test_format_results_with_scores():
    """Scores are rendered when provided."""
    row = _row(id="test:1")
    result = format_results([row], scores={"test:1": 0.876})
    assert "(score: 0.876)" in result


def test_format_results_no_score_for_row():
    """Row without a matching score omits the score label."""
    row = _row(id="test:1")
    result = format_results([row], scores={"other:2": 0.5})
    assert "score:" not in result


def test_format_results_multiple():
    """Multiple results are separated by double newlines."""
    rows = [_row(id="a", type_name="Alpha"), _row(id="b", type_name="Beta")]
    result = format_results(rows)
    assert "Type: Alpha" in result
    assert "Type: Beta" in result
    # Two results separated by blank line
    assert result.count("---") == 4  # 2 results x 2 dashes each


def test_format_results_empty_optional_fields():
    """Fields that are empty strings are omitted from the header."""
    row = _row(module_path="", category="", heading="", type_name="")
    result = format_results([row])
    assert "Module:" not in result
    assert "Category:" not in result
    assert "Section:" not in result
    assert "Type:" not in result
    # Source and File should still be present
    assert "Source: test" in result
    assert "File: test.py" in result


# ---------------------------------------------------------------------------
# _escape_like tests
# ---------------------------------------------------------------------------


def test_escape_like_percent():
    """Percent sign is escaped for LIKE queries."""
    assert _escape_like("100%") == "100\\%"


def test_escape_like_underscore():
    """Underscore is escaped for LIKE queries."""
    assert _escape_like("my_func") == "my\\_func"


def test_escape_like_backslash():
    """Backslash itself is escaped first."""
    assert _escape_like("a\\b") == "a\\\\b"


def test_escape_like_clean():
    """String without wildcards passes through unchanged."""
    assert _escape_like("MyClass") == "MyClass"


# ---------------------------------------------------------------------------
# _sanitize_fts tests
# ---------------------------------------------------------------------------


def test_sanitize_fts_strips_quotes():
    """Double quotes are removed (would break FTS5 phrase syntax)."""
    assert _sanitize_fts('hello"world') == "helloworld"


def test_sanitize_fts_strips_star():
    """Asterisk is removed (FTS5 prefix/suffix operator)."""
    assert _sanitize_fts("test*") == "test"


def test_sanitize_fts_strips_caret():
    """Caret is removed (FTS5 initial token operator)."""
    assert _sanitize_fts("^first") == "first"


def test_sanitize_fts_strips_leading_plus_minus():
    """Leading +/- are stripped (FTS5 required/excluded operators)."""
    assert _sanitize_fts("+required") == "required"
    assert _sanitize_fts("-excluded") == "excluded"


def test_sanitize_fts_normal_text_unchanged():
    """Normal text passes through unchanged."""
    assert _sanitize_fts("MyClass") == "MyClass"


def test_sanitize_fts_empty_after_strip():
    """String that becomes empty after sanitization returns empty."""
    assert _sanitize_fts('"*"') == ""


# ---------------------------------------------------------------------------
# _confidence_tier tests
# ---------------------------------------------------------------------------


def test_confidence_tier_high():
    """Score at or above high threshold is HIGH."""
    assert _confidence_tier(0.90, 0.85, 0.65) == "HIGH"
    assert _confidence_tier(0.85, 0.85, 0.65) == "HIGH"


def test_confidence_tier_medium():
    """Score between medium and high thresholds is MEDIUM."""
    assert _confidence_tier(0.75, 0.85, 0.65) == "MEDIUM"
    assert _confidence_tier(0.65, 0.85, 0.65) == "MEDIUM"


def test_confidence_tier_low():
    """Score below medium threshold is LOW."""
    assert _confidence_tier(0.50, 0.85, 0.65) == "LOW"
    assert _confidence_tier(0.64, 0.85, 0.65) == "LOW"


# ---------------------------------------------------------------------------
# format_results with confidence and gotcha
# ---------------------------------------------------------------------------


def test_format_results_with_confidence():
    """Confidence tier is shown alongside score."""
    row = _row(id="test:1")
    result = format_results(
        [row],
        scores={"test:1": 0.876},
        confidence={"test:1": "HIGH"},
    )
    assert "score: 0.876, confidence: HIGH" in result


def test_format_results_with_gotcha():
    """Gotcha text is appended as CAUTION."""
    row = _row(id="test:1", gotcha="Not a timeout — DNS failure")
    result = format_results([row])
    assert "[CAUTION: Not a timeout — DNS failure]" in result


def test_format_results_empty_gotcha_no_caution():
    """Empty gotcha field does not produce a CAUTION line."""
    row = _row(id="test:1", gotcha="")
    result = format_results([row])
    assert "CAUTION" not in result


def test_format_results_no_gotcha_key():
    """Row without gotcha key does not produce a CAUTION line."""
    row = _row(id="test:1")
    # DictRow without 'gotcha' key
    result = format_results([row])
    assert "CAUTION" not in result


# ---------------------------------------------------------------------------
# _get_gotcha tests
# ---------------------------------------------------------------------------


def test_get_gotcha_present():
    """Returns gotcha text when key exists."""
    row = DictRow({"gotcha": "warning text"})
    assert _get_gotcha(row) == "warning text"


def test_get_gotcha_missing_key():
    """Returns empty string when key doesn't exist."""
    row = DictRow({"id": "test:1"})
    assert _get_gotcha(row) == ""


def test_get_gotcha_none_value():
    """Returns empty string when value is None."""
    row = DictRow({"gotcha": None})
    assert _get_gotcha(row) == ""
