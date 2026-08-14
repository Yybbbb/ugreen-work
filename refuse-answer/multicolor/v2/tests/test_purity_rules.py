from segment_color_import import add_repo_path

add_repo_path()

from purity_rules import decide_purity


def test_confident_pure_passes():
    result = decide_purity(
        main_ratio=0.86,
        second_ratio=0.07,
        minor_total_ratio=0.14,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is True
    assert result["is_multicolor"] is False
    assert result["decision_reason"] == "pure"


def test_boundary_minor_total_020_rejects():
    result = decide_purity(
        main_ratio=0.80,
        second_ratio=0.09,
        minor_total_ratio=0.20,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "minor_total_ratio>=0.20"


def test_single_strong_secondary_rejects():
    result = decide_purity(
        main_ratio=0.84,
        second_ratio=0.11,
        minor_total_ratio=0.16,
        n_effective_colors=2,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "second_ratio>=0.10"


def test_multiple_small_colors_reject():
    result = decide_purity(
        main_ratio=0.82,
        second_ratio=0.09,
        minor_total_ratio=0.18,
        n_effective_colors=3,
        neutral_l_multipeak=False,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "n_effective_colors>=3 and minor_total_ratio>=0.15"


def test_neutral_l_multipeak_rejects():
    result = decide_purity(
        main_ratio=0.91,
        second_ratio=0.05,
        minor_total_ratio=0.09,
        n_effective_colors=2,
        neutral_l_multipeak=True,
    )
    assert result["is_pure"] is False
    assert result["decision_reason"] == "neutral_l_multipeak"
