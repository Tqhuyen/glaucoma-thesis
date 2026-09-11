import ast
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts import final_data as fd
from scripts import final_training as ft


@pytest.fixture
def prepared(tmp_path):
    root, cache = tmp_path / "data", tmp_path / "cache"
    root.mkdir()
    cache.mkdir()
    np.save(root / "Training_volumes.npy", np.arange(3 * 4**3, dtype=np.uint8).reshape(3, 4, 4, 4))
    np.save(root / "Training_labels.npy", np.array([0, 1, 0], dtype=np.int64))
    calls = []

    def denoise(volume):
        calls.append(volume.shape)
        return volume // 2

    options = dict(res2d=3, denoise_fn=denoise, method="bilateral", params={"diameter": 5}, implementation="test-v1")
    return root, cache, options, calls


def prepare(prepared, **overrides):
    root, cache, options, _ = prepared
    return fd.prepare_data(root, cache, ["Training"], **{**options, "smoke": True, "publish_cache": True, **overrides})


def no_work(*args, **kwargs):
    pytest.fail("Unexpected expensive work or copy")


def test_missing_no_build_and_unknown_file_are_retained(prepared, monkeypatch):
    root, cache, _, calls = prepared
    monkeypatch.setattr(ft, "build_views", no_work)
    with pytest.raises(RuntimeError, match="ALLOW_BUILD_DENOISED"):
        prepare(prepared)
    assert calls == []
    assert not list(cache.iterdir())
    target = root / "Training_volumes_dn.npy"
    shutil.copyfile(root / "Training_volumes.npy", target)
    before = target.read_bytes()
    with pytest.raises(RuntimeError, match="Unverified"):
        prepare(prepared, allow_build=True)
    assert target.read_bytes() == before


def test_relocation_rebinds_markers_and_final_dataset(prepared, tmp_path, monkeypatch):
    root, cache, options, calls = prepared
    manifest = prepare(prepared, allow_build=True)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for kind in ("volumes", "labels"):
        path = relocated / f"Training_{kind}.npy"
        shutil.copyfile(root / path.name, path)
        os.utime(path, ns=(1_000_000_000, 1_000_000_000))
    monkeypatch.setattr(ft, "build_denoised", no_work)
    original_views = ft.build_views
    monkeypatch.setattr(ft, "build_views", no_work)
    restored = fd.prepare_data(relocated, cache, ["Training"], **options, smoke=True)
    assert restored == manifest
    assert len(calls) == 3
    target = relocated / "Training_volumes_dn.npy"
    marker = torch.load(target.with_suffix(".complete.pt"), weights_only=False)
    assert marker["source"] == ft.file_identity(relocated / "Training_volumes.npy")
    view_marker = torch.load(relocated / "Training_volumes_dn_views_3.complete.pt", weights_only=False)
    assert view_marker["source"] == ft.file_identity(target)
    monkeypatch.setattr(ft, "build_views", original_views)
    ds = ft.FinalDataset(target, relocated / "Training_labels.npy", res3d=4, res2d=3, seed=42)
    assert len(ds) == 3
    assert ds[0][0].shape == (1, 4, 4, 4)
    assert not list(cache.rglob("Training_volumes.npy"))


@pytest.mark.parametrize("kind", ["denoised", "labels", "views", "dzs"])
def test_remote_content_corruption_rejected(prepared, tmp_path, kind):
    root, cache, options, calls = prepared
    prepare(prepared, allow_build=True)
    remote = next(cache.iterdir())
    name = fd._names("Training", 3)[kind]
    array = np.load(remote / name).copy()
    array.flat[0] += 1
    np.save(remote / name, array)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for suffix in ("volumes", "labels"):
        shutil.copyfile(root / f"Training_{suffix}.npy", relocated / f"Training_{suffix}.npy")
    with pytest.raises(RuntimeError, match="No valid complete cache"):
        fd.prepare_data(relocated, cache, ["Training"], **options, smoke=True)
    assert len(calls) == 3


def test_local_corruption_restores_remote_not_legacy_marker(prepared):
    root, _, _, calls = prepared
    manifest = prepare(prepared, allow_build=True)
    path = root / "Training_volumes_dn_views_3.npy"
    np.save(path, np.zeros((3, 2, 3, 3), dtype=np.uint8))
    assert prepare(prepared) == manifest
    assert fd._array_identity(path) == manifest["splits"]["Training"]["files"]["views"]
    assert len(calls) == 3


@pytest.mark.parametrize("override", [{"params": {"diameter": 7}}, {"implementation": "test-v2"}, {"res2d": 5}])
def test_parameter_mismatch_no_build(prepared, override):
    prepare(prepared, allow_build=True)
    with pytest.raises(RuntimeError, match="No valid complete cache"):
        prepare(prepared, **override)
    assert len(prepared[3]) == 3


@pytest.mark.parametrize("suffix", ["volumes", "labels"])
def test_raw_content_change_rejected_even_with_local_markers(prepared, suffix):
    root, _, _, _ = prepared
    prepare(prepared, allow_build=True)
    path = root / f"Training_{suffix}.npy"
    array = np.load(path).copy()
    array.flat[0] += 1
    np.save(path, array)
    with pytest.raises(RuntimeError, match="No valid complete cache"):
        prepare(prepared)


def test_partial_recovery_does_not_repeat_denoise(prepared):
    root, cache, _, calls = prepared
    for count in (1, 2):
        with pytest.raises(RuntimeError, match="pending"):
            prepare(prepared, allow_build=True, limit=1)
        assert len(calls) == count
        assert (root / "Training_volumes_dn.partial.npy").exists()
        assert not list(cache.rglob("manifest.json"))
        with pytest.raises(RuntimeError, match="pending"):
            prepare(prepared)
    manifest = prepare(prepared, allow_build=True, limit=1)
    assert len(calls) == 3
    assert "Training" in manifest["splits"]
    assert not (root / "Training_volumes_dn.partial.npy").exists()


def test_view_failure_recovers_without_repeating_denoise(prepared, monkeypatch):
    original = ft.build_views

    def fail(*args, **kwargs):
        raise OSError("interrupted views")

    monkeypatch.setattr(ft, "build_views", fail)
    with pytest.raises(OSError, match="interrupted"):
        prepare(prepared, allow_build=True)
    monkeypatch.setattr(ft, "build_views", original)
    prepare(prepared, allow_build=True)
    assert len(prepared[3]) == 3


def test_sync_skips_copy_and_cached_hash_then_handles_metadata_change(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.bin", tmp_path / "destination.bin"
    source.write_bytes(b"verified bytes")
    assert fd.sync_verified(source, destination)
    monkeypatch.setattr(fd.shutil, "copyfile", no_work)
    original = fd.hashlib.sha256
    monkeypatch.setattr(fd.hashlib, "sha256", no_work)
    assert not fd.sync_verified(source, destination)
    monkeypatch.setattr(fd.hashlib, "sha256", original)
    stat = destination.stat()
    os.utime(destination, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    assert not fd.sync_verified(source, destination)


def test_copy_corruption_keeps_old_destination(tmp_path, monkeypatch):
    source, destination = tmp_path / "source.bin", tmp_path / "destination.bin"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    def corrupt(source, target):
        target.write_bytes(b"bad")

    monkeypatch.setattr(fd.shutil, "copyfile", corrupt)
    with pytest.raises(RuntimeError, match="Copy verification"):
        fd.sync_verified(source, destination)
    assert destination.read_bytes() == b"old"
    assert not list(tmp_path.glob("*.tmp-*"))


def test_manifest_covers_actual_content_shape_dtype_and_versions(prepared):
    root, cache, _, _ = prepared
    manifest = prepare(prepared, allow_build=True)
    entry = manifest["splits"]["Training"]
    names = fd._names("Training", 3)
    for kind, identity in {**entry["sources"], **entry["files"]}.items():
        path = root / names[kind]
        array = np.load(path, mmap_mode="r")
        assert identity == dict(
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size=path.stat().st_size,
            shape=list(array.shape),
            dtype=array.dtype.str,
        )
    assert manifest["config"]["versions"]["code_sha256"]
    assert json.loads(next(cache.rglob("manifest.json")).read_text()) == manifest
    assert str(root) not in json.dumps(manifest)


def test_legacy_complete_local_markers_can_be_adopted(prepared, monkeypatch):
    root, _, options, calls = prepared
    target = root / "Training_volumes_dn.npy"
    ft.build_denoised(
        root / "Training_volumes.npy",
        target,
        options["denoise_fn"],
        method=options["method"],
        params=options["params"],
        implementation=options["implementation"],
    )
    ft.build_views(target, res2d=3)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    prepare(prepared)
    assert len(calls) == 3


def test_space_precheck_before_denoising(prepared, monkeypatch):
    usage = shutil.disk_usage(prepared[0])
    monkeypatch.setattr(fd.shutil, "disk_usage", lambda _: type(usage)(usage.total, usage.total, 0))
    with pytest.raises(RuntimeError, match="Insufficient"):
        prepare(prepared, allow_build=True)
    assert prepared[3] == []


def test_repeat_preparation_does_not_upload_or_build(prepared, monkeypatch):
    manifest = prepare(prepared, allow_build=True)
    monkeypatch.setattr(fd.shutil, "copyfile", no_work)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    assert prepare(prepared) == manifest


def test_upload_failure_recovers_local_complete_cache(prepared, monkeypatch):
    original = fd.shutil.copyfile

    def fail(*args, **kwargs):
        raise OSError("Drive unavailable")

    monkeypatch.setattr(fd.shutil, "copyfile", fail)
    with pytest.raises(OSError, match="Drive unavailable"):
        prepare(prepared, allow_build=True)
    monkeypatch.setattr(fd.shutil, "copyfile", original)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    prepare(prepared)
    assert len(prepared[3]) == 3


def test_no_build_preflights_every_split_before_restoring(prepared, tmp_path, monkeypatch):
    root, cache, options, _ = prepared
    prepare(prepared, allow_build=True)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for split in ("Training", "Test"):
        for suffix in ("volumes", "labels"):
            shutil.copyfile(root / f"Training_{suffix}.npy", relocated / f"{split}_{suffix}.npy")
    monkeypatch.setattr(fd.shutil, "copyfile", no_work)
    with pytest.raises(RuntimeError, match="No valid complete cache for Test"):
        fd.prepare_data(relocated, cache, ["Training", "Test"], **options, smoke=True)
    assert not (relocated / "Training_volumes_dn.npy").exists()


def test_subset_preparation_preserves_other_manifest_entries(prepared):
    root, cache, options, _ = prepared
    prepare(prepared, allow_build=True)
    for suffix in ("volumes", "labels"):
        shutil.copyfile(root / f"Training_{suffix}.npy", root / f"Test_{suffix}.npy")
    manifest = fd.prepare_data(root, cache, ["Test"], **options, allow_build=True, smoke=True)
    assert set(manifest["splits"]) == {"Training", "Test"}
    assert prepare(prepared) == manifest


def test_view_algorithm_mismatch_cannot_downgrade_to_legacy(prepared, monkeypatch):
    prepare(prepared, allow_build=True)
    monkeypatch.setitem(fd._VIEW_SIGNATURE, "code_sha256", "changed-view-algorithm")
    with pytest.raises(RuntimeError, match="No valid complete cache"):
        prepare(prepared)


def test_partial_raw_content_change_is_rejected(prepared):
    root, _, _, calls = prepared
    with pytest.raises(RuntimeError, match="pending"):
        prepare(prepared, allow_build=True, limit=1)
    path = root / "Training_volumes.npy"
    array = np.load(path).copy()
    array.flat[0] += 1
    np.save(path, array)
    with pytest.raises(RuntimeError, match="Partial content/parameter identity mismatch"):
        prepare(prepared, allow_build=True)
    assert len(calls) == 1


@pytest.mark.parametrize("kind", ["views", "dzs"])
def test_shape_dtype_validated_even_when_manifest_hash_matches(prepared, tmp_path, kind):
    root, cache, options, _ = prepared
    prepare(prepared, allow_build=True)
    manifest_path = next(cache.rglob("manifest.json"))
    path = manifest_path.parent / fd._names("Training", 3)[kind]
    array = np.zeros((3, 1, 3, 3), dtype=np.uint8) if kind == "views" else np.zeros(3, dtype=np.int64)
    np.save(path, array)
    manifest = json.loads(manifest_path.read_text())
    manifest["splits"]["Training"]["files"][kind] = fd._array_identity(path)
    fd._json_write(manifest_path, manifest)
    relocated = tmp_path / "relocated"
    relocated.mkdir()
    for suffix in ("volumes", "labels"):
        shutil.copyfile(root / f"Training_{suffix}.npy", relocated / f"Training_{suffix}.npy")
    with pytest.raises(RuntimeError, match="No valid complete cache"):
        fd.prepare_data(relocated, cache, ["Training"], **options, smoke=True)


@pytest.mark.parametrize("explicit", [False, True])
def test_cross_torch_producer_versions_restore_without_build_or_publish(prepared, tmp_path, monkeypatch, explicit):
    root, cache, options, _ = prepared
    original = prepare(prepared, allow_build=True)
    selected = next(cache.rglob("manifest.json"))
    relocated = tmp_path / "gpu-data"
    relocated.mkdir()
    for suffix in ("volumes", "labels"):
        shutil.copyfile(root / f"Training_{suffix}.npy", relocated / f"Training_{suffix}.npy")
    monkeypatch.setitem(fd._VERSIONS, "torch", "99.0.0+cu999")
    monkeypatch.setitem(fd._VERSIONS, "code_sha256", "different-orchestration-code")
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    copy = fd.sync_verified

    def only_download(source, destination):
        assert not Path(destination).is_relative_to(cache)
        return copy(source, destination)

    monkeypatch.setattr(fd, "sync_verified", only_download)
    restored = fd.prepare_data(
        relocated,
        cache,
        ["Training"],
        **options,
        smoke=True,
        manifest_path=selected if explicit else None,
    )
    assert restored == original


def test_same_implementation_name_but_different_actual_filter_rejected(prepared):
    prepare(prepared, allow_build=True)

    def different(volume):
        return volume // 3

    with pytest.raises(RuntimeError, match="No valid complete cache"):
        prepare(prepared, denoise_fn=different)
    with pytest.raises(ValueError, match="incompatible"):
        prepare(prepared, denoise_fn=different, manifest_path=next(prepared[1].rglob("manifest.json")))


def test_ambiguous_outputs_require_explicit_manifest(prepared):
    root, cache, _, _ = prepared
    prepare(prepared, allow_build=True)
    first = next(cache.iterdir())
    second = cache / "alternative-producer"
    shutil.copytree(first, second)
    path = second / "Training_volumes_dn_views_3.npy"
    array = np.load(path).copy()
    array.flat[0] += 1
    np.save(path, array)
    manifest = json.loads((second / "manifest.json").read_text())
    manifest["splits"]["Training"]["files"]["views"] = fd._array_identity(path)
    fd._json_write(second / "manifest.json", manifest)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        prepare(prepared, publish_cache=False)
    restored = prepare(prepared, manifest_path=second / "manifest.json", publish_cache=False)
    assert fd._array_identity(root / path.name) == restored["splits"]["Training"]["files"]["views"]


def test_build_and_legacy_adoption_do_not_publish_by_default(prepared, monkeypatch):
    root, cache, options, _ = prepared
    monkeypatch.setattr(fd, "sync_verified", no_work)
    fd.prepare_data(root, cache, ["Training"], **options, allow_build=True, smoke=True)
    assert not list(cache.iterdir())
    shutil.rmtree(root / ".final_data")
    fd.prepare_data(root, cache, ["Training"], **options, smoke=True)
    assert not list(cache.iterdir())


@pytest.fixture
def cpu_export(prepared, tmp_path):
    root, cache, _, _ = prepared
    export = tmp_path / "cpu-export"
    export.mkdir()
    raw = fd._array_identity(root / "Training_volumes.npy")
    labels = fd._array_identity(root / "Training_labels.npy")
    np.save(export / "Training_volumes_dn.npy", np.load(root / "Training_volumes.npy") // 2)
    shutil.copyfile(root / "Training_labels.npy", export / "Training_labels.npy")
    revision = "a" * 40
    repo = "owner/raw-oct"
    sources = {
        "Training": {
            "shape": raw["shape"],
            "dtype": "uint8",
            "offset": raw["size"] - 3 * 4**3,
            "sample_bytes": 4**3,
            "size": raw["size"],
            "filename": "Training_volumes.npy",
            "repo_revision": revision,
            "labels_sha256": labels["sha256"],
            "class_counts": {"0": 2, "1": 1},
            "url": f"https://huggingface.co/datasets/{repo}/resolve/{revision}/Training_volumes.npy",
        }
    }
    identity = {
        "source_repo": repo,
        "source_revision": revision,
        "sources": sources,
        "method": "bilateral",
        "params": fd._CPU_PARAMS.copy(),
        "axis": 0,
        "implementation": fd._CPU_IMPLEMENTATION,
        "filter_code_sha256": fd._CPU_FILTER_HASH,
        "skimage": "0.24.0",
        "numpy": "1.26.0",
        "output_dtype": "uint8",
        "resolution": 200,
        "quantization": fd._CPU_QUANTIZATION,
        "format": 1,
    }
    marker = {
        "identity": identity,
        "shape": raw["shape"],
        "dtype": "uint8",
        "file": "Training_volumes_dn.npy",
        "sha256": fd._array_identity(export / "Training_volumes_dn.npy")["sha256"],
        "complete": True,
    }
    fd._json_write(export / "Training_complete.json", marker)
    fd._json_write(
        export / "manifest.json",
        {
            "identity": identity,
            "complete": True,
            "total_volumes": 3,
            "splits": {"Training": marker},
        },
    )
    fd._json_write(
        root / "source_manifest.json",
        {
            "source_repo": repo,
            "source_revision": revision,
            "sources": {"Training": {"sha256": raw["sha256"], "labels_sha256": labels["sha256"]}},
        },
    )
    options = dict(res2d=3, method="bilateral", params=fd._CPU_PARAMS.copy(), implementation=fd._CPU_IMPLEMENTATION)
    return root, export, cache, options


def test_known_export_code_matches_exporter_without_import_side_effects():
    exporter = Path(fd.__file__).with_name("prepare_bilateral_200.py")
    if not exporter.is_file():
        pytest.skip("Optional external CPU exporter is not installed; import format is tested with fixtures")
    source = exporter.read_text(encoding="utf-8")
    params = next(
        node.value
        for node in ast.parse(source).body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "PARAMS" for target in node.targets)
    )
    assert ast.literal_eval(params) == fd._CPU_PARAMS
    function = next(
        node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == "bilateral"
    )
    actual = ast.get_source_segment(source, function) + "\n"
    assert fd._code_hash(actual) == fd._CPU_FILTER_HASH
    assert fd._code_hash(actual.replace("\n", "\r\n")) == fd._CPU_FILTER_HASH


def test_cpu_export_ingestion_builds_views_once_never_denoises_or_publishes(cpu_export, monkeypatch):
    root, export, cache, options = cpu_export
    monkeypatch.setattr(ft, "build_denoised", no_work)
    original = ft.build_views
    calls = []

    def views(source, **kwargs):
        calls.append(Path(source))
        return original(source, **kwargs)

    monkeypatch.setattr(ft, "build_views", views)
    manifest = fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
    assert len(calls) == 1
    assert not list(cache.iterdir())
    assert manifest["splits"]["Training"]["provenance"]["cpu_export"]["source_revision"] == "a" * 40
    fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
    assert len(calls) == 1
    assert not list(cache.iterdir())
    marker = torch.load(root / "Training_volumes_dn.complete.pt", weights_only=False)
    assert marker["source"] == ft.file_identity(root / "Training_volumes.npy")
    fd.prepare_data(root, cache, ["Training"], **options, denoise_fn=None, smoke=True)
    assert len(calls) == 1


def test_cpu_export_explicit_publication_then_gpu_restore(cpu_export, tmp_path, monkeypatch):
    root, export, cache, options = cpu_export
    manifest = fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True, allow_publish=True)
    assert len(list(cache.rglob("manifest.json"))) == 1
    assert not list(cache.rglob("Training_volumes.npy"))
    relocated = tmp_path / "gpu"
    relocated.mkdir()
    for suffix in ("volumes", "labels"):
        shutil.copyfile(root / f"Training_{suffix}.npy", relocated / f"Training_{suffix}.npy")
    monkeypatch.setitem(fd._VERSIONS, "torch", "999.0+cuda")
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    restored = fd.prepare_data(relocated, cache, ["Training"], **options, denoise_fn=None, smoke=True)
    assert restored == manifest


@pytest.mark.parametrize("bad", ["params", "code", "axis", "revision", "raw_hash", "labels", "output", "marker"])
def test_cpu_export_refuses_mismatches_before_work(cpu_export, monkeypatch, bad):
    root, export, cache, options = cpu_export
    manifest = json.loads((export / "manifest.json").read_text())
    raw_manifest = json.loads((root / "source_manifest.json").read_text())
    if bad == "params":
        manifest["identity"]["params"]["sigma_color"] = 0.2
    elif bad == "code":
        manifest["identity"]["filter_code_sha256"] = "0" * 64
    elif bad == "axis":
        manifest["identity"]["axis"] = 1
    elif bad == "revision":
        raw_manifest["source_revision"] = "b" * 40
    elif bad == "raw_hash":
        raw_manifest["sources"]["Training"]["sha256"] = "0" * 64
    elif bad in ("labels", "output"):
        name = "Training_labels.npy" if bad == "labels" else "Training_volumes_dn.npy"
        array = np.load(export / name).copy()
        array.flat[0] += 1
        np.save(export / name, array)
    elif bad == "marker":
        manifest["splits"]["Training"]["complete"] = False
    fd._json_write(export / "manifest.json", manifest)
    fd._json_write(root / "source_manifest.json", raw_manifest)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    monkeypatch.setattr(fd, "sync_verified", no_work)
    with pytest.raises(ValueError):
        fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
    assert not list(cache.iterdir())


def test_cpu_import_requires_trusted_download_provenance(cpu_export, monkeypatch):
    root, export, cache, options = cpu_export
    source_path = root / "source_manifest.json"
    alternate = root / "pinned_download.json"
    source_path.rename(alternate)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    with pytest.raises(ValueError, match="Trusted raw source manifest"):
        fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
    fd.import_cpu_export(
        root,
        export,
        cache,
        ["Training"],
        **options,
        smoke=True,
        source_manifest_path=alternate,
    )


def test_cpu_import_retries_keep_original_view_provenance_and_skip_copies(cpu_export, monkeypatch):
    root, export, cache, options = cpu_export
    first = fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True, allow_publish=True)
    monkeypatch.setitem(fd._VERSIONS, "torch", "new-consumer")
    monkeypatch.setattr(fd.shutil, "copyfile", no_work)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    monkeypatch.setattr(ft, "build_views", no_work)
    second = fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True, allow_publish=True)
    assert second == first
    assert len(list(cache.rglob("manifest.json"))) == 1


def test_bad_explicit_manifest_never_falls_back_to_build(prepared, monkeypatch):
    prepare(prepared, allow_build=True)
    path = next(prepared[1].rglob("manifest.json"))
    manifest = json.loads(path.read_text())
    manifest["splits"]["Training"]["files"]["denoised"]["sha256"] = "0" * 64
    fd._json_write(path, manifest)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    with pytest.raises(RuntimeError, match="Selected manifest"):
        prepare(prepared, allow_build=True, manifest_path=path)


def test_publication_flags_require_real_booleans(prepared, cpu_export):
    with pytest.raises(ValueError, match="explicit booleans"):
        prepare(prepared, publish_cache="False")
    root, export, cache, options = cpu_export
    with pytest.raises(ValueError, match="explicit booleans"):
        fd.import_cpu_export(root, export, cache, ["Training"], **options, allow_publish="False")


def test_cpu_import_missing_explicit_filter_default_is_not_guessed(cpu_export, monkeypatch):
    root, export, cache, options = cpu_export
    options["params"].pop("win_size")
    monkeypatch.setattr(fd, "sync_verified", no_work)
    with pytest.raises(ValueError, match="explicit effective parameters"):
        fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)


def test_cpu_import_recovers_after_views_completed_before_manifest(cpu_export, monkeypatch):
    root, export, cache, options = cpu_export
    original = ft.build_views

    def interrupted(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("after views")

    monkeypatch.setattr(ft, "build_views", interrupted)
    monkeypatch.setattr(ft, "build_denoised", no_work)
    with pytest.raises(OSError, match="after views"):
        fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
    monkeypatch.setattr(ft, "build_views", no_work)
    monkeypatch.setattr(fd.shutil, "copyfile", no_work)
    fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)


def test_cpu_import_accepts_only_line_ending_variation_of_known_code(cpu_export):
    root, export, cache, options = cpu_export
    manifest = json.loads((export / "manifest.json").read_text())
    windows_hash = hashlib.sha256(fd._CPU_FILTER_SOURCE.replace("\n", "\r\n").encode()).hexdigest()
    manifest["identity"]["filter_code_sha256"] = windows_hash
    manifest["splits"]["Training"]["identity"]["filter_code_sha256"] = windows_hash
    fd._json_write(export / "manifest.json", manifest)
    fd._json_write(export / "Training_complete.json", manifest["splits"]["Training"])
    fd.import_cpu_export(root, export, cache, ["Training"], **options, smoke=True)
