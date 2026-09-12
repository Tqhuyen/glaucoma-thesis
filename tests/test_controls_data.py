import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from scripts import controls_data as cd


def metadata(path, *, lfs=False):
    content = path.read_bytes()
    return SimpleNamespace(
        rfilename=path.name,
        size=len(content),
        lfs={"sha256": hashlib.sha256(content).hexdigest()} if lfs else None,
        blob_id=hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest(),
    )


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    import huggingface_hub

    raw, remote, data, export = [tmp_path / name for name in ("raw-hf", "export-hf", "data", "export")]
    for path in (raw, remote, data, export):
        path.mkdir()
    np.save(raw / "Training_volumes.npy", np.arange(16, dtype=np.uint8).reshape(2, 2, 2, 2))
    np.save(raw / "Training_labels.npy", np.array([0, 1], dtype=np.int64))
    for path in raw.iterdir():
        shutil.copyfile(path, data / path.name)
    np.save(remote / "Training_volumes_dn.npy", np.zeros((2, 2, 2, 2), dtype=np.uint8))
    shutil.copyfile(raw / "Training_labels.npy", remote / "Training_labels.npy")
    source_revision = "a" * 40
    identity = {
        "source_repo": "owner/raw",
        "source_revision": source_revision,
        "resolution": 200,
        "sources": {
            "Training": {
                "shape": [2, 200, 200, 200],
                "repo_revision": source_revision,
                "labels_sha256": hashlib.sha256((raw / "Training_labels.npy").read_bytes()).hexdigest(),
            }
        },
    }
    marker = {"identity": identity, "complete": True, "file": "Training_volumes_dn.npy"}
    manifest = {"complete": True, "identity": identity, "splits": {"Training": marker}}
    cd.fd._json_write(remote / "manifest.json", manifest)
    calls = []
    raw_info = SimpleNamespace(
        sha=source_revision,
        siblings=[metadata(path, lfs=path.name.endswith("_volumes.npy")) for path in raw.iterdir()],
    )

    def dataset_info(repo, *, revision, files_metadata):
        assert files_metadata is True
        calls.append(("info", repo, revision))
        if repo == "owner/raw":
            assert revision == source_revision
            return raw_info
        assert repo == cd.DEFAULT_REPO and revision == cd.DEFAULT_REVISION
        return SimpleNamespace(sha=revision, siblings=[metadata(p) for p in remote.iterdir()])

    def snapshot_download(**kwargs):
        assert kwargs["token"] == "test-secret"
        assert kwargs["repo_type"] == "dataset"
        patterns = kwargs["allow_patterns"]
        assert all("*" not in name for name in patterns)
        calls.append(("download", kwargs))
        source = raw if kwargs["repo_id"] == "owner/raw" else remote
        assert kwargs["revision"] == (source_revision if source == raw else cd.DEFAULT_REVISION)
        target = Path(kwargs["local_dir"])
        for name in patterns:
            shutil.copyfile(source / name, target / name)
        return str(target)

    api = Mock(dataset_info=Mock(side_effect=dataset_info))
    constructor = Mock(return_value=api)
    monkeypatch.setattr(huggingface_hub, "HfApi", constructor)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    monkeypatch.setattr(cd.fd.ft, "load_env_file", Mock(side_effect=AssertionError("No .env reads")))
    monkeypatch.setattr(cd.fd.ft, "build_denoised", Mock(side_effect=AssertionError("No denoising")))
    importer = Mock(return_value={"config": {"res2d": 96}, "splits": {"Training": {}}})
    monkeypatch.setattr(cd.fd, "import_cpu_export", importer)
    return SimpleNamespace(
        raw=raw,
        remote=remote,
        data=data,
        export=export,
        manifest=manifest,
        raw_info=raw_info,
        calls=calls,
        api=constructor,
        importer=importer,
        options=dict(token="test-secret", splits=["Training"], res2d=96, store_res=200),
    )


def run(prepared, **overrides):
    return cd.prepare_bilateral(prepared.data, prepared.export, **{**prepared.options, **overrides})


def test_verified_local_raw_selective_export_and_import_config(prepared, capsys):
    p = prepared
    result = run(p)
    p.api.assert_called_once_with(token="test-secret")
    downloads = [call[1] for call in p.calls if call[0] == "download"]
    assert [call["allow_patterns"] for call in downloads] == [
        ["manifest.json"],
        ["Training_volumes_dn.npy", "Training_labels.npy"],
    ]
    assert all(call["repo_id"] == cd.DEFAULT_REPO for call in downloads)
    trusted = json.loads((p.data / "source_manifest.json").read_text())
    assert trusted["source_repo"] == "owner/raw"
    assert trusted["source_revision"] == "a" * 40
    assert trusted["sources"]["Training"] == {
        "sha256": hashlib.sha256((p.raw / "Training_volumes.npy").read_bytes()).hexdigest(),
        "labels_sha256": hashlib.sha256((p.raw / "Training_labels.npy").read_bytes()).hexdigest(),
    }
    assert json.loads((p.export / "Training_complete.json").read_text()) == p.manifest["splits"]["Training"]
    p.importer.assert_called_once_with(
        p.data,
        p.export,
        p.data / ".controls_cache",
        ["Training"],
        res2d=96,
        method="bilateral",
        params=cd.fd._CPU_PARAMS,
        implementation=cd.fd._CPU_IMPLEMENTATION,
        smoke=False,
        allow_publish=False,
        source_manifest_path=p.data / "source_manifest.json",
    )
    assert result["manifest"] == p.importer.return_value
    assert result["raw_download"] == trusted
    assert result["store_res"] == 200 and result["res2d"] == 96
    assert "test-secret" not in json.dumps(result) + capsys.readouterr().out


def test_only_missing_raw_downloaded_and_reverified(prepared):
    p = prepared
    (p.data / "Training_labels.npy").unlink()
    run(p)
    raw_downloads = [c[1] for c in p.calls if c[0] == "download" and c[1]["repo_id"] == "owner/raw"]
    assert len(raw_downloads) == 1
    assert raw_downloads[0]["allow_patterns"] == ["Training_labels.npy"]
    assert raw_downloads[0]["local_dir"] == str(p.data)


@pytest.mark.parametrize("token", [None, "", "  ", True])
def test_auth_required_before_hf(prepared, token):
    with pytest.raises(ValueError, match="token"):
        run(prepared, token=token)
    prepared.api.assert_not_called()


@pytest.mark.parametrize(
    "override",
    [{"revision": "main"}, {"store_res": 96}, {"res2d": 0}, {"splits": []}, {"splits": ["../Training"]}],
)
def test_invalid_config_before_hf(prepared, override):
    with pytest.raises(ValueError):
        run(prepared, **override)
    prepared.api.assert_not_called()


@pytest.mark.parametrize("name", ["Training_volumes.npy", "Training_labels.npy"])
def test_wrong_local_bytes_fail_without_replacement_or_provenance(prepared, name):
    p = prepared
    path = p.data / name
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="checksum mismatch"):
        run(p)
    assert path.read_bytes() == content
    assert not (p.data / "source_manifest.json").exists()
    assert not any(c[0] == "download" and c[1]["repo_id"] == "owner/raw" for c in p.calls)
    p.importer.assert_not_called()


@pytest.mark.parametrize("bad", ["incomplete", "source_revision", "split_revision", "shape", "labels", "raw_hash"])
def test_bad_export_fails_before_provenance(prepared, bad):
    p = prepared
    identity = p.manifest["identity"]
    source = identity["sources"]["Training"]
    if bad == "incomplete":
        p.manifest["complete"] = False
    elif bad == "source_revision":
        identity["source_revision"] = "main"
    elif bad == "split_revision":
        source["repo_revision"] = "b" * 40
    elif bad == "shape":
        source["shape"] = [2, 96, 96, 96]
    elif bad == "labels":
        source["labels_sha256"] = "0" * 64
    else:
        source["sha256"] = "0" * 64
    cd.fd._json_write(p.remote / "manifest.json", p.manifest)
    with pytest.raises(ValueError):
        run(p)
    assert not (p.data / "source_manifest.json").exists()
    p.importer.assert_not_called()


@pytest.mark.parametrize("bad", ["revision", "missing", "lfs", "blob", "size"])
def test_untrusted_hf_metadata_fails(prepared, bad):
    p = prepared
    raw = next(f for f in p.raw_info.siblings if f.rfilename.endswith("_volumes.npy"))
    labels = next(f for f in p.raw_info.siblings if f.rfilename.endswith("_labels.npy"))
    if bad == "revision":
        p.raw_info.sha = "b" * 40
    elif bad == "missing":
        p.raw_info.siblings = []
    elif bad == "lfs":
        raw.lfs = None
    elif bad == "blob":
        labels.blob_id = "0" * 40
    else:
        raw.size += 1
    with pytest.raises(ValueError):
        run(p)
    p.importer.assert_not_called()
    assert not (p.data / "source_manifest.json").exists()


@pytest.mark.parametrize("content", ["{broken", '{"complete": false}'])
def test_corrupt_existing_completion_not_replaced(prepared, content):
    p = prepared
    path = p.export / "Training_complete.json"
    path.write_text(content)
    cd.fd._json_write(p.remote / path.name, p.manifest["splits"]["Training"])
    with pytest.raises(ValueError, match="Corrupt completion"):
        run(p)
    assert path.read_text() == content
    p.importer.assert_not_called()


def test_downloads_available_completion_marker(prepared):
    p = prepared
    cd.fd._json_write(p.remote / "Training_complete.json", p.manifest["splits"]["Training"])
    run(p)
    assert "Training_complete.json" in [c[1] for c in p.calls if c[0] == "download"][-1]["allow_patterns"]


def test_existing_provenance_revision_cannot_be_relabelled(prepared):
    p = prepared
    cd.fd._json_write(p.data / "source_manifest.json", {"source_repo": "owner/raw", "source_revision": "b" * 40})
    with pytest.raises(ValueError, match="provenance mismatch"):
        run(p)
    assert json.loads((p.data / "source_manifest.json").read_text())["source_revision"] == "b" * 40
    p.importer.assert_not_called()


def test_rerun_still_hashes_raw_and_propagates_import_failure(prepared):
    p = prepared
    run(p)
    p.importer.side_effect = ValueError("CPU export output/labels checksum or shape mismatch")
    with pytest.raises(ValueError, match="CPU export"):
        run(p)
    assert p.importer.call_count == 2
    path = p.data / "Training_volumes.npy"
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="Pinned HF checksum"):
        run(p)
    assert p.importer.call_count == 2


def test_stream_hash_does_not_use_read_bytes(prepared, monkeypatch):
    monkeypatch.setattr(Path, "read_bytes", Mock(side_effect=AssertionError("Must stream")))
    path = prepared.data / "Training_volumes.npy"
    remote = next(f for f in prepared.raw_info.siblings if f.rfilename == path.name)
    assert cd._verify_file(path, remote) == remote.lfs["sha256"]
