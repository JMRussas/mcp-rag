#  mcp-rag - Server Tests
#
#  Tests for result formatting, LIKE escaping, and search filtering.

from server import _escape_like, _sanitize_fts, format_results

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
