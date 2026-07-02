import pytest

from caspoc import manual_page, manual_topics


def test_manual_topics_are_available():
    assert {"CASPOC", "CASPOC.outputs", "CASPOC.R", "SparsePLS"}.issubset(
        manual_topics()
    )


def test_manual_page_returns_table():
    page = manual_page("CASPOC")

    assert list(page.columns) == ["section", "name", "description"]
    assert "correct use" in set(page["section"])
    assert "tune before test" in set(page["name"])


def test_r_manual_mapping_is_available():
    page = manual_page("CASPOC.R")

    assert "name mapping" in set(page["section"])
    assert "ncomp -> n_components" in set(page["name"])


def test_manual_page_rejects_unknown_topic():
    with pytest.raises(ValueError, match="Unknown manual topic"):
        manual_page("missing")
