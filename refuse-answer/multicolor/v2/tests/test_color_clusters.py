from segment_color_import import add_repo_path

add_repo_path()

from color_clusters import compute_cluster_stats_from_centers


def test_ratios_use_pixel_area_not_peak_height():
    lab = (
        [[0.0, 60.0, 40.0] for _ in range(80)]
        + [[0.0, -50.0, 35.0] for _ in range(10)]
        + [[0.0, 20.0, -60.0] for _ in range(10)]
    )
    centers = [[60.0, 40.0], [-50.0, 35.0], [20.0, -60.0]]

    stats = compute_cluster_stats_from_centers(lab, centers)

    assert stats["main_ratio"] == 0.80
    assert stats["second_ratio"] == 0.10
    assert stats["minor_total_ratio"] == 0.20
    assert stats["n_effective_colors"] == 3
