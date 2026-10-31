"""Test to verify that all metrics reported in README.md match docs/evaluation.md exactly.

Fails if README.md numbers diverge from docs/evaluation.md.
"""

from scripts.build.render_readme_metrics import (
    END_TAG,
    EVALUATION_PATH,
    README_PATH,
    START_TAG,
    generate_metrics_markdown_table,
    parse_evaluation_metrics,
)


def test_readme_metrics_match_evaluation_md():
    """Verify that README.md's metrics table exactly matches docs/evaluation.md."""
    assert EVALUATION_PATH.exists(), f"Missing {EVALUATION_PATH}"
    assert README_PATH.exists(), f"Missing {README_PATH}"

    eval_text = EVALUATION_PATH.read_text(encoding="utf-8")
    readme_text = README_PATH.read_text(encoding="utf-8")

    # 1. Parse metrics from docs/evaluation.md
    metrics = parse_evaluation_metrics(eval_text)

    # Assert essential metrics are properly extracted and non-empty
    assert "survival_model_c_idx" in metrics
    assert "not_planned_model_prauc" in metrics
    assert "label_macro_f1" in metrics
    assert "loco_mean_f1" in metrics
    assert "loco_pooled_f1" in metrics
    assert "vol_mae" in metrics

    # 2. Generate the expected markdown table
    expected_table = generate_metrics_markdown_table(metrics)

    # 3. Assert START_TAG and END_TAG exist in README
    assert START_TAG in readme_text, f"Missing {START_TAG} in README.md"
    assert END_TAG in readme_text, f"Missing {END_TAG} in README.md"

    # 4. Extract current table from README
    start_idx = readme_text.index(START_TAG) + len(START_TAG)
    end_idx = readme_text.index(END_TAG)
    current_table = readme_text[start_idx:end_idx].strip()

    # 5. Assert equality of normalized tables
    assert current_table == expected_table.strip(), (
        "Metrics in README.md differ from docs/evaluation.md!\n"
        f"Expected:\n{expected_table.strip()}\n\n"
        f"Found in README.md:\n{current_table}\n\n"
        "Run 'python scripts/build/render_readme_metrics.py' to synchronize."
    )


def test_readme_contains_specific_verified_numbers():
    """Explicitly assert key empirical numbers appear in README.md and favourable-only numbers are absent."""
    readme_text = README_PATH.read_text(encoding="utf-8")

    # Assert exact required out-of-sample & conservative numbers from docs/evaluation.md
    assert "0.5426" in readme_text, "Missing Survival C-Index 0.5426 in README.md"
    assert "0.514" in readme_text, "Missing Survival Out-of-Sample C-Index 0.514 in README.md"
    assert "[0.492, 0.536]" in readme_text, "Missing Survival Out-of-Sample CI [0.492, 0.536] in README.md"

    assert "0.222" in readme_text, "Missing Not-Planned Offline PR-AUC 0.222 in README.md"
    assert "0.195" in readme_text, "Missing Not-Planned Out-of-Sample PR-AUC 0.195 in README.md"
    assert "0.163" in readme_text, "Missing Prevalence Baseline 0.163 in README.md"

    assert "0.31 – 1.00" in readme_text, "Missing LOCO per-fold range 0.31 – 1.00 in README.md"
    assert "0.62" in readme_text, "Missing LOCO Conservative Mean F1 0.62 in README.md"
    assert "0.51" in readme_text, "Missing LOCO Conservative Pooled F1 0.51 in README.md"
    assert "optimistic, 17 positives, no unseen data left" in readme_text, "Missing honesty caveat in README.md"

    assert "0.644" in readme_text, "Missing Label Out-of-Sample Macro-F1 0.644 in README.md"
    assert "76.86" in readme_text, "Missing ETS MAE 76.86 in README.md"

    # Strictly fail on favourable-only ungrounded numbers
    start_idx = readme_text.index(START_TAG)
    end_idx = readme_text.index(END_TAG)
    table_text = readme_text[start_idx:end_idx]

    assert "0.281" not in table_text, "Favourable-only PR-AUC 0.281 found in README metrics table!"
    assert "0.595 [0.50, 0.70]" not in table_text, "Favourable-only Pooled F1 0.595 [0.50, 0.70] found in README table!"

    # Assert calibration findings are accurately stated and "strong monotonic calibration" is absent
    assert "lowest bin predicts 0.11 vs observed 0.34" in readme_text or "lowest bin 0.11 vs 0.34" in readme_text, (
        "Missing 14-day calibration under-prediction disclosure in README.md"
    )
    assert "top bin predicts 0.86 vs observed 0.59" in readme_text or "0.86 predicted vs 0.59 observed" in readme_text, (
        "Missing 30-day calibration over-confidence disclosure in README.md"
    )
    assert "strong monotonic calibration" not in readme_text, (
        "Inaccurate 'strong monotonic calibration' claim found in README.md!"
    )


def test_hf_space_metrics_match_evaluation_md():
    """Assert HF Space demo numbers come strictly from docs/evaluation.md and fail on favourable-only numbers."""
    from pathlib import Path
    space_html_path = Path("hf_space/index.html")
    space_readme_path = Path("hf_space/README.md")
    assert space_html_path.exists(), "Missing hf_space/index.html"
    assert space_readme_path.exists(), "Missing hf_space/README.md"

    html_text = space_html_path.read_text(encoding="utf-8")
    readme_text = space_readme_path.read_text(encoding="utf-8")

    # Assert exact required metrics in index.html
    assert "0.5426" in html_text, "Missing Survival C-Index 0.5426 in HF Space"
    assert "0.195" in html_text, "Missing Not-Planned Out-of-Sample PR-AUC 0.195 in HF Space"
    assert "0.145" in html_text, "Missing Test Prevalence 0.145 in HF Space"
    assert "0.610" in html_text, "Missing Not-Planned ROC-AUC 0.610 in HF Space"
    assert "80.0%" in html_text, "Missing Label Top-1 80.0% in HF Space"
    assert "76.72%" in html_text, "Missing Label always-other baseline 76.72% in HF Space"
    assert "0.644" in html_text, "Missing Label Macro-F1 0.644 in HF Space"
    assert "0.31 – 1.00" in html_text, "Missing LOCO per-fold range 0.31 – 1.00 in HF Space"
    assert "0.62" in html_text, "Missing LOCO Conservative Mean F1 0.62 in HF Space"
    assert "0.51" in html_text, "Missing LOCO Conservative Pooled F1 0.51 in HF Space"
    assert "Optimistic, 17 positives, no unseen data left" in html_text, "Missing honesty disclosure in HF Space"

    # Strictly fail on favourable-only ungrounded numbers in HF Space
    assert "0.281" not in html_text, "Favourable-only PR-AUC 0.281 found in HF Space index.html!"
    assert "0.595 [0.50, 0.70]" not in html_text, "Favourable-only Pooled F1 0.595 [0.50, 0.70] found in HF Space index.html!"
    assert "0.281" not in readme_text, "Favourable-only PR-AUC 0.281 found in HF Space README.md!"

    # Assert accurate calibration findings in HF Space
    assert "lowest bin 0.11 vs 0.34" in html_text, (
        "Missing 14-day calibration under-prediction disclosure in HF Space index.html"
    )
    assert "0.86 predicted vs 0.59 observed" in html_text, (
        "Missing 30-day calibration over-confidence disclosure in HF Space index.html"
    )
    assert "strong monotonic calibration" not in html_text, (
        "Inaccurate 'strong monotonic calibration' claim found in HF Space index.html!"
    )
    assert "strong monotonic calibration" not in readme_text, (
        "Inaccurate 'strong monotonic calibration' claim found in HF Space README.md!"
    )

