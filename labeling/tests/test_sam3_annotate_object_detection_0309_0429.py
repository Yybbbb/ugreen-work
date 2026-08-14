import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "sam3_annotate_object_detection_0309_0429.py"
)


def load_script_module():
    spec = importlib.util.spec_from_file_location("sam3_object_detection", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_iter_images_streams_paths_without_materializing(tmp_path):
    module = load_script_module()
    image_dir = tmp_path / "images"
    scene_dir = image_dir / "scene_a"
    scene_dir.mkdir(parents=True)
    (scene_dir / "a.jpg").write_bytes(b"a")
    (scene_dir / "b.txt").write_text("skip", encoding="utf-8")

    iterator = module.iter_images(image_dir=image_dir)

    assert not isinstance(iterator, list)
    assert list(iterator) == [scene_dir / "a.jpg"]


def test_output_path_preserves_scene_directory(tmp_path):
    module = load_script_module()
    image_dir = tmp_path / "images"
    out_dir = tmp_path / "annotations"
    image_path = image_dir / "2026_4_29_park" / "sample_p00.jpg"

    assert module.output_path(image_path, image_dir, out_dir) == (
        out_dir / "2026_4_29_park" / "sample_p00.json"
    )


def test_build_annotation_uses_dataset_relative_paths(tmp_path):
    module = load_script_module()
    image_dir = tmp_path / "images"
    image_path = image_dir / "2026_4_29_park" / "sample_p00.jpg"
    attrs = {name: module.empty_attr() for name in module.CLASSES}

    annotation = module.build_annotation(
        image_path=image_path,
        image_dir=image_dir,
        attributes=attrs,
        server="http://localhost:8010",
        elapsed=0.12,
        img_width=64,
        img_height=128,
    )

    assert annotation["sample_id"] == "object_detection_0309_0429/2026_4_29_park/sample_p00"
    assert annotation["source"]["scene"] == "2026_4_29_park"
    assert annotation["image"]["path"] == "images/2026_4_29_park/sample_p00.jpg"
    assert annotation["annotation_path"] == "annotations/2026_4_29_park/sample_p00.json"

