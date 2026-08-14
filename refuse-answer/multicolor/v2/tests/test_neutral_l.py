from segment_color_import import add_repo_path

add_repo_path()

from neutral_l import analyze_neutral_l


def test_black_white_neutral_split_detected():
    lab = [[5.0, 0.0, 0.0] for _ in range(500)] + [[95.0, 0.0, 0.0] for _ in range(500)]
    result = analyze_neutral_l(lab)
    assert result["neutral_l_multipeak"] is True
    assert result["l_second_ratio"] >= 0.10


def test_chromatic_region_skips_l_split():
    lab = [[35.0, 60.0, 35.0] for _ in range(500)] + [[80.0, 60.0, 35.0] for _ in range(500)]
    result = analyze_neutral_l(lab)
    assert result["neutral_l_multipeak"] is False
    assert result["skipped"] is True
