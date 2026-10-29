"""Test verifying linked tickets for sandbox comment #8."""

from scripts.evaluation.verify_sandbox_links import verify_similar_links


def test_verify_sandbox_links_returns_valid_records():
    records = verify_similar_links()
    assert len(records) == 3

    # Check each target ticket #181690, #179464, #178581
    rec_by_num = {r["ticket_number"]: r for r in records}
    assert 181690 in rec_by_num
    assert 179464 in rec_by_num
    assert 178581 in rec_by_num

    for num in [181690, 179464, 178581]:
        rec = rec_by_num[num]
        assert rec["same_integration"] is True, f"Ticket #{num} is not same integration 'zha'"
        assert rec["ge_threshold"] is True, f"Ticket #{num} similarity {rec['similarity_score']} < 0.70"
        assert rec["similarity_score"] >= 0.70
        assert rec["redirect_url"] == f"https://redirect.github.com/home-assistant/core/issues/{num}"
        assert rec["formatted_link"].startswith(f"[{rec['title']} (home-assistant/core#{num})]")
        assert f"(home-assistant/core#{num})" in rec["formatted_link"]


def test_verify_sandbox_links_fixture_fallback():
    records = verify_similar_links(issues_path="non_existent_data.parquet")
    assert len(records) == 3
    for r in records:
        assert r["same_integration"] is True
        assert r["similarity_score"] >= 0.70
        assert "home-assistant/core#" in r["formatted_link"]
