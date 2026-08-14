from segment_color_import import add_repo_path

add_repo_path()

from color_space import chroma_ab, rgb_to_lab


def test_rgb_to_lab_shape_and_range():
    rgb = [[[0, 0, 0], [255, 255, 255], [255, 0, 0]]]
    lab = rgb_to_lab(rgb)
    assert len(lab) == 1
    assert len(lab[0]) == 3
    assert len(lab[0][0]) == 3
    assert lab[0][0][0] <= 1.0
    assert lab[0][1][0] >= 99.0
    assert lab[0][2][1] > 40.0


def test_chroma_ab_black_white_is_low():
    rgb = [[0, 0, 0], [255, 255, 255], [128, 128, 128]]
    lab = rgb_to_lab(rgb)
    c = chroma_ab(lab)
    assert all(value < 2.0 for value in c)
