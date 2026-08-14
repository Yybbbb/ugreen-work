from segment_color_import import add_repo_path

add_repo_path()

from detector_v4 import detect_region


def test_red_shadow_region_is_pure():
    img = []
    for _ in range(20):
        row = [[50, 0, 0] for _ in range(10)] + [[215, 150, 150] for _ in range(10)]
        img.append(row)
    mask = [[1 for _ in range(20)] for _ in range(20)]
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is True
    assert result["shadow_merges"] >= 1


def test_black_white_region_is_multicolor():
    img = []
    for _ in range(20):
        row = [[0, 0, 0] for _ in range(10)] + [[255, 255, 255] for _ in range(10)]
        img.append(row)
    mask = [[1 for _ in range(20)] for _ in range(20)]
    result = detect_region(img, mask, 1, "upper")
    assert result["is_pure"] is False
    assert result["decision_reason"] == "second_ratio>=0.10"


def test_sampled_checkerboard_still_detects_multicolor():
    img = []
    for y in range(80):
        row = []
        for x in range(80):
            row.append([20, 20, 20] if (x + y) % 2 == 0 else [245, 245, 245])
        img.append(row)
    mask = [[1 for _ in range(80)] for _ in range(80)]

    result = detect_region(img, mask, 1, "upper")

    assert result["n_pixels"] == 6400
    assert result["sampled_pixels"] < result["n_pixels"]
    assert result["is_pure"] is False
    assert result["second_ratio"] >= 0.10
