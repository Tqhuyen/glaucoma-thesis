import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import run_denoise_full_cuda as job


class Response:
    def __init__(self, body, status=200, headers=None, fail=False):
        self.body, self.status_code, self.headers, self.fail = body, status, headers or {}, fail

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise OSError("HTTP failure")

    def iter_content(self, size):
        yield self.body
        if self.fail:
            raise OSError("Disconnected")


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = iter(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


@pytest.fixture(autouse=True)
def no_wait(monkeypatch):
    monkeypatch.setattr(job.time, "sleep", lambda _: None)
    monkeypatch.setattr(job, "disk_gate", lambda *args: None)


def test_download_resume_strict_authenticated(tmp_path):
    target = tmp_path / "raw.npy"
    session = Session(Response(b"abc", fail=True), Response(b"def", 206, {"Content-Range": "bytes 3-5/6"}))
    job.download_verified(
        session, "https://example.invalid/raw", "secret", target, 6, hashlib.sha256(b"abcdef").hexdigest()
    )
    assert target.read_bytes() == b"abcdef"
    assert session.calls[1][1]["headers"]["Range"] == "bytes=3-5"
    assert session.calls[0][1]["headers"]["Authorization"] == "Bearer secret"
    assert not target.with_name("raw.npy.partial").exists()


@pytest.mark.parametrize("response", [Response(b"def", 200), Response(b"def", 206, {"Content-Range": "bytes 0-2/6"})])
def test_resume_rejects_wrong_range(tmp_path, response):
    target = tmp_path / "raw.npy"
    partial = tmp_path / "raw.npy.partial"
    partial.write_bytes(b"abc")
    with pytest.raises(RuntimeError, match="partial preserved"):
        job.download_verified(Session(response), "https://example.invalid", "secret", target, 6, "x", attempts=1)
    assert partial.read_bytes() == b"abc"
    assert not target.exists()


def test_raw_cache_requires_hash_and_never_overwrites(tmp_path):
    target = tmp_path / "raw.npy"
    target.write_bytes(b"existing")
    session = Session()
    job.download_verified(session, "unused", "secret", target, 8, job.digest(target))
    assert not session.calls
    with pytest.raises(ValueError, match="preserved"):
        job.download_verified(session, "unused", "secret", target, 8, "wrong")
    assert target.read_bytes() == b"existing"


def archive_with(session):
    archive = object.__new__(job.Archive)
    archive.token, archive.session = "secret", session
    return archive


def test_remote_verify_pinned_full_hash():
    archive = archive_with(Session(Response(b"abcdef")))
    receipt = {
        "remote": "classical/tv/config/volumes/Test/shard-00000.npy",
        "commit": "a" * 40,
        "size": 6,
        "sha256": hashlib.sha256(b"abcdef").hexdigest(),
    }
    archive.verify(receipt)
    url, kwargs = archive.session.calls[0]
    assert receipt["commit"] in url
    assert kwargs["headers"]["Authorization"] == "Bearer secret"


def test_upload_failure_retains_volume(tmp_path):
    target = tmp_path / "shard-00000.npy"
    target.write_bytes(b"abcdef")
    archive = archive_with(Session(*(Response(b"wrong") for _ in range(6))))
    archive.api = SimpleNamespace(
        dataset_info=lambda _: SimpleNamespace(private=True), upload_file=lambda **kwargs: SimpleNamespace(oid="a" * 40)
    )
    with pytest.raises(RuntimeError, match="no cleanup"):
        archive.upload(target, "classical/tv/config/volumes/Test/shard-00000.npy")
    assert target.read_bytes() == b"abcdef"


def receipt_for(target):
    return {
        "verified": True,
        "generated": str(target.resolve()),
        "artifacts": [
            {
                "remote": f"volumes/Training/{target.name}",
                "sha256": job.digest(target),
                "size": target.stat().st_size,
                "commit": "a" * 40,
            }
        ],
    }


def test_cleanup_exact_allowlist(tmp_path):
    root = tmp_path / "shards"
    root.mkdir()
    target = root / "shard-00000.npy"
    target.write_bytes(b"generated")
    original = tmp_path / "Training_volumes_dn.npy"
    original.write_bytes(b"original")
    receipt = receipt_for(target)
    with pytest.raises(ValueError):
        job.cleanup_generated(original, root, receipt)
    with pytest.raises(ValueError):
        job.cleanup_generated(target, root, {**receipt, "verified": False})
    job.cleanup_generated(target, root, receipt)
    assert not target.exists()
    assert original.read_bytes() == b"original"


def test_resume_verification_failure_prevents_cleanup(tmp_path):
    target = tmp_path / "shard-00000.npy"
    target.write_bytes(b"generated")
    marker = tmp_path / "receipt.json"
    job.atomic_json(marker, receipt_for(target))

    def fail(_):
        raise OSError("Remote missing")

    with pytest.raises(OSError):
        job.resume_shard(marker, SimpleNamespace(verify=fail), tmp_path)
    assert target.exists()
    calls = []
    assert job.resume_shard(marker, SimpleNamespace(verify=calls.append), tmp_path)
    assert len(calls) == 1 and not target.exists()
    assert job.resume_shard(marker, SimpleNamespace(verify=calls.append), tmp_path)


def test_mutated_generated_file_is_not_deleted(tmp_path):
    target = tmp_path / "shard-00000.npy"
    target.write_bytes(b"generated")
    receipt = receipt_for(target)
    target.write_bytes(b"different")
    with pytest.raises(ValueError, match="changed"):
        job.cleanup_generated(target, tmp_path, receipt)
    assert target.exists()


def test_bilateral_hash_checked_before_load(tmp_path):
    manifest = {
        "complete": True,
        "identity": {"source_repo": job.SOURCE, "source_revision": job.REVISION},
        "splits": {"Training": {"complete": True, "sha256": "wrong"}},
    }
    job.atomic_json(tmp_path / "manifest.json", manifest)
    original = tmp_path / "Training_volumes_dn.npy"
    original.write_bytes(b"keep")
    with pytest.raises(ValueError, match="Bilateral SHA256 mismatch"):
        job.load_bilateral(tmp_path, {})
    assert original.read_bytes() == b"keep"


def test_config_id_ignores_device_and_validation_timing():
    meta = {"method": "tv", "parameters": {"weight": 0.1}, "device": "A", "validated": False}
    changed = {**meta, "device": "B", "validated": True, "validation": {"seconds": 999}}
    assert job.config_id(meta, {}, {}) == job.config_id(changed, {}, {})
    assert job.config_id(meta, {}, {}) != job.config_id({**meta, "parameters": {"weight": 0.2}}, {}, {})


def test_cli_defaults_and_pending_methods():
    args = job.parser().parse_args(["--execute"])
    assert args.methods == ["bilateral", "bm3d", "gaussian", "median", "tv"]
    assert set(job.METHODS) - set(args.methods) == {"wavelet", "nlm"}
    assert args.max_compute_hours == 4
    assert job.parser().parse_args(["--pilot-only", "--methods", "tv"]).pilot_only


def test_missing_token_no_output_creation(tmp_path, monkeypatch):
    from pipeline import utils

    monkeypatch.setattr(utils, "load_env_file", lambda: None)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    target = tmp_path / "new"
    with pytest.raises(SystemExit, match="HF_TOKEN required"):
        job.main(["--execute", "--work-dir", str(target)])
    assert not target.exists()


def test_missing_wandb_key_never_prompts(tmp_path, monkeypatch):
    from pipeline import utils

    monkeypatch.setattr(utils, "load_env_file", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="no interactive login"):
        job.main(["--execute", "--work-dir", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_aggregate_small_global_records(tmp_path):
    directory = tmp_path / "metrics" / "Training"
    directory.mkdir(parents=True)
    records = [
        {"split": "Training", "class": 0, "index": i, "SNR": value, "invalid_reasons": {}}
        for i, value in enumerate([1.0, None, 3.0])
    ]
    (directory / "shard-00000.jsonl").write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    import pandas as pd

    frame = pd.read_csv(job.aggregate(tmp_path))
    row = frame.iloc[0]
    assert row["count"] == 2 and row["invalid_count"] == 1
    assert row["mean"] == 2 and row["std_ddof1"] == pytest.approx(np.sqrt(2))
    assert row["p25"] == 1.5 and row["p75"] == 2.5


def test_aggregate_retains_entirely_undefined_metric(tmp_path):
    directory = tmp_path / "metrics" / "Test"
    directory.mkdir(parents=True)
    record = {"split": "Test", "class": 1, "index": 0, "beta": None, "invalid_reasons": {"beta": "constant"}}
    (directory / "shard-00000.jsonl").write_text(json.dumps(record), encoding="utf-8")
    import pandas as pd

    row = pd.read_csv(job.aggregate(tmp_path)).iloc[0]
    assert row["metric"] == "beta" and row["count"] == 0 and row["invalid_count"] == 1
    assert np.isnan(row["mean"])


def test_wandb_failure_is_fatal():
    def fail(_):
        raise OSError("Disconnected")

    with pytest.raises(job.FatalJobError):
        job.log_online(SimpleNamespace(log=fail), {"metric": 1})


@pytest.mark.parametrize("fail_upload", [False, True])
def test_shard_transaction_and_completed_resume(tmp_path, monkeypatch, fail_upload):
    monkeypatch.setattr(job, "CASES", ())
    monkeypatch.setattr(job, "wait_gpu", lambda *args: None)
    calls, verified = [], []

    class Context:
        def __init__(self, raw):
            pass

        def evaluate(self, den):
            return {"SNR": 1.0, "invalid_reasons": {}}, [{"slice_index": 0, "SNR": 1.0}]

    def backend(raw):
        calls.append(1)
        return raw

    def upload(path, remote):
        if fail_upload and remote.endswith(".npy"):
            raise job.FatalJobError("Upload failed")
        return {"remote": remote, "sha256": job.digest(path), "size": path.stat().st_size, "commit": "a" * 40}

    archive = SimpleNamespace(upload=upload, verify=verified.append, remote_completed=lambda _: None)
    args = SimpleNamespace(work_dir=tmp_path, drive_dir=None, wait_gpu_seconds=0)
    arrays = {"Training": (np.zeros((2, 200, 200, 200), dtype=np.uint8), np.array([0, 1]))}
    suite = SimpleNamespace(MetricContext=Context, metric_protocol={"version": "test"}, release_memory=lambda: None)
    cp = SimpleNamespace(asnumpy=np.asarray)
    run = SimpleNamespace(log=lambda _: None, log_artifact=lambda _: None)
    baseline = tmp_path / "original" / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "per_volume.csv").write_text("split,index,class,SNR\nTraining,0,0,1.0\nTraining,1,1,1.0\n")
    parameters = (
        args,
        "gaussian",
        backend,
        {"method": "gaussian"},
        arrays,
        None,
        archive,
        run,
        {"methods": {"original": {"config_id": "baseline"}}},
        lambda **kwargs: None,
        lambda: False,
        cp,
        suite,
    )
    if fail_upload:
        with pytest.raises(job.FatalJobError):
            job.run_method(*parameters)
        assert len(list(tmp_path.rglob("shard-*.npy"))) == 1
        assert not list(tmp_path.rglob("*.receipt.json"))
        return
    result = job.run_method(*parameters)
    assert result["status"] == "complete"
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.npy"))
    assert len(list(tmp_path.rglob("*.slices.jsonl.gz"))) == 1
    import pandas as pd

    aggregate = pd.read_csv(next(tmp_path.rglob("aggregate.csv")))
    assert set(aggregate["scope"]) == {"split_class", "split", "whole_dataset"}
    paired = pd.read_csv(next(tmp_path.rglob("paired_vs_raw.csv")))
    assert paired["SNR_minus_raw"].eq(0).all()
    result = job.run_method(*parameters)
    assert result["resumed_remote"]
    assert len(calls) == 2 and len(verified) >= 6


def test_full_oct_validation_checks_all_eight_planes():
    class Backend:
        name = "original"
        cp = SimpleNamespace(asnumpy=np.asarray)

        def __call__(self, raw):
            assert raw.shape == (8, 200, 200)
            result = raw.copy()
            result[7, 199, 199] = 1
            return result

    assert not job.validate_oct_planes(Backend(), np.zeros((200, 200, 200), np.uint8))["passed"]


def test_existing_public_repository_refused(monkeypatch):
    import huggingface_hub

    api = SimpleNamespace(whoami=lambda: {"name": "owner"}, dataset_info=lambda _: SimpleNamespace(private=False))
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    with pytest.raises(RuntimeError, match="already public"):
        job.Archive("secret")


def test_remote_completion_checks_every_shard():
    artifact = {
        "remote": "classical/tv/id/volumes/Test/shard-00000.npy",
        "commit": "a" * 40,
        "size": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
    }
    manifest = {"complete": True, "shards": [{"artifacts": [artifact]}]}
    body = json.dumps(manifest).encode()
    manifest_receipt = {
        "remote": "classical/tv/id/manifest.json",
        "commit": "a" * 40,
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    marker = {"complete": True, "volumes": 3300, "manifest": manifest_receipt, "reports": [manifest_receipt]}
    archive = archive_with(Session(Response(json.dumps(marker).encode()), Response(body)))
    archive.api = SimpleNamespace(
        dataset_info=lambda _: SimpleNamespace(sha="b" * 40), file_exists=lambda *a, **k: True
    )
    calls = []
    archive.verify = calls.append
    recovered, receipt = archive.remote_completed("classical/tv/id")
    assert recovered == manifest and receipt["remote_only"]
    assert artifact in calls
    assert "b" * 40 in archive.session.calls[0][0]
    assert "a" * 40 in archive.session.calls[1][0]
