#  mcp-rag - Base Chunker Tests
#
#  Tests for shared utilities in chunkers/base.py (brace counting, category derivation).

from chunkers.base import count_braces, derive_category

# ---------------------------------------------------------------------------
# Brace counting tests
# ---------------------------------------------------------------------------


def test_count_braces_simple():
    """Basic brace counting without comments or strings."""
    assert count_braces("{ }")[0] == 0
    assert count_braces("{")[0] == 1
    assert count_braces("}")[0] == -1
    assert count_braces("{ { }")[0] == 1


def test_count_braces_skips_line_comment():
    """Braces in // comments are ignored."""
    assert count_braces("code // { not counted }")[0] == 0
    assert count_braces("{ // }")[0] == 1


def test_count_braces_skips_block_comment():
    """Braces in /* */ comments are ignored."""
    delta, still_in = count_braces("code /* { */")
    assert delta == 0
    assert not still_in

    # Multi-line block comment
    delta, still_in = count_braces("code /* start {")
    assert delta == 0
    assert still_in

    delta, still_in = count_braces("still in comment } */", in_block_comment=True)
    assert delta == 0
    assert not still_in


def test_count_braces_skips_string_literals():
    """Braces inside string literals are ignored."""
    assert count_braces('var x = "{ }";')[0] == 0
    assert count_braces('var x = "{{}}";')[0] == 0


def test_count_braces_skips_char_literal():
    """Braces in char literals are ignored."""
    assert count_braces("var c = '{';")[0] == 0
    assert count_braces("var c = '}';")[0] == 0


def test_count_braces_verbatim_string():
    """Braces in C# verbatim strings (@"...") are ignored."""
    assert count_braces('@"{ }"')[0] == 0
    assert count_braces('@"test ""quoted"" { }"')[0] == 0


def test_count_braces_hash_comment():
    """Braces after # comment prefix are ignored."""
    delta, _ = count_braces("# comment { here }", line_comment="#")
    assert delta == 0

    delta, _ = count_braces("code { # comment }", line_comment="#")
    assert delta == 1  # only the first { counts


def test_count_braces_real_csharp():
    """Realistic C# lines parse correctly."""
    assert count_braces("public class Foo {")[0] == 1
    assert count_braces("    var dict = new Dictionary<string, string> { };")[0] == 0
    assert count_braces("}")[0] == -1
    assert count_braces('    if (x) { return "hello {world}"; }')[0] == 0


# ---------------------------------------------------------------------------
# derive_category tests
# ---------------------------------------------------------------------------


def test_derive_category_nested():
    """Nested path uses parent directory as category."""
    assert derive_category("src/utils/helpers.py") == "utils"


def test_derive_category_single_file():
    """Single file with no directory returns 'core'."""
    assert derive_category("helpers.py") == "core"


def test_derive_category_shallow():
    """Shallow path uses first parent."""
    assert derive_category("src/main.py") == "src"
