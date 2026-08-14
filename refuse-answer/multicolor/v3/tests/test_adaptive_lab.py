from segment_color_import import add_repo_path

add_repo_path()

from adaptive_lab import adaptive_lab_distance, cluster_lab_pixels


def test_neutral_black_white_distance_is_large():
    black = [5.0, 0.0, 0.0]
    white = [95.0, 0.0, 0.0]
    assert adaptive_lab_distance(black, white) > 40.0


def test_chromatic_shadow_distance_is_small():
    dark_red = [25.0, 55.0, 35.0]
    bright_red = [80.0, 56.0, 36.0]
    assert adaptive_lab_distance(dark_red, bright_red) < 18.0


def test_chromatic_different_colors_distance_is_large():
    red = [50.0, 60.0, 40.0]
    blue = [50.0, 20.0, -60.0]
    assert adaptive_lab_distance(red, blue) > 40.0


def test_cluster_merges_same_color_shadow():
    pixels = [[25.0, 55.0, 35.0] for _ in range(50)] + [[80.0, 56.0, 36.0] for _ in range(50)]
    stats = cluster_lab_pixels(pixels)
    assert stats["n_effective_colors"] == 1
    assert stats["main_ratio"] == 1.0


def test_cluster_splits_black_and_white():
    pixels = [[5.0, 0.0, 0.0] for _ in range(50)] + [[95.0, 0.0, 0.0] for _ in range(50)]
    stats = cluster_lab_pixels(pixels)
    assert stats["n_effective_colors"] == 2
    assert stats["second_ratio"] == 0.5
