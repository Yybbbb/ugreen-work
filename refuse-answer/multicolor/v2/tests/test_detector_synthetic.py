from segment_color_import import add_repo_path

add_repo_path()

from detector import detect_region


def test_pure_red_region_passes():
    img = [[[220, 20, 20] for _ in range(20)] for _ in range(20)]
    mask = [[1 for _ in range(20)] for _ in range(20)]
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is True


def test_black_white_split_rejects():
    img = [[[0, 0, 0] if x < 10 else [255, 255, 255] for x in range(20)] for _ in range(20)]
    mask = [[1 for _ in range(20)] for _ in range(20)]
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is False
    assert result["neutral_l"]["neutral_l_multipeak"] is True


def test_main_color_with_three_minor_colors_rejects():
    row = (
        [[220, 20, 20] for _ in range(76)]
        + [[20, 220, 20] for _ in range(8)]
        + [[20, 20, 220] for _ in range(8)]
        + [[240, 220, 20] for _ in range(8)]
    )
    img = [list(row) for _ in range(10)]
    mask = [[1 for _ in range(100)] for _ in range(10)]
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is False
    assert result["minor_total_ratio"] >= 0.20
