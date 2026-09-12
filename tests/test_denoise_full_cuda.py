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


def test_private_storage_quota_does_not_retry_or_delete(tmp_path):
    target = tmp_path / "shard-00000.npy"
    target.write_bytes(b"keep pending bytes")
    calls = []

    class QuotaFailure(Exception):
        response = SimpleNamespace(json=lambda: {"error": "Private repository storage limit reached, please upgrade"})

    def commit(**kwargs):
        calls.append(kwargs)
        raise QuotaFailure("Bad request: Private")

    archive = archive_with(Session())
    archive.api = SimpleNamespace(create_commit=commit)
    with pytest.raises(job.StorageQuotaError, match="private repository storage limit"):
        archive.upload_many([(target, "classical/test/shard-00000.npy")], "resume")
    assert len(calls) == 1
    assert target.read_bytes() == b"keep pending bytes"
    assert not list(tmp_path.glob("*.receipt.json"))


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
    archive.token, archive.session, archive._index = "secret", session, None
    archive._visibility_at = float("inf")
    return archive


def test_remote_verify_pinned_full_hash():
    archive = archive_with(Session(Response(b"abcdef")))
    archive._index = {}
    archive.api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(sha=k["revision"], siblings=[]))
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


def test_remote_verify_uses_server_hash_without_download():
    archive = archive_with(Session())
    remote = "classical/tv/config/volumes/Test/shard-00000.npy"
    archive._index = {
        "a" * 40: {remote: {"size": 6, "expected": {"kind": "sha256", "value": hashlib.sha256(b"abcdef").hexdigest()}}}
    }
    archive.api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(siblings=[]))
    archive.verify({"remote": remote, "commit": "a" * 40, "size": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()})
    assert archive.session.calls == []


def test_remote_verify_server_hash_mismatch_blocks():
    archive = archive_with(Session())
    remote = "classical/tv/config/volumes/Test/shard-00000.npy"
    archive._index = {"a" * 40: {remote: {"size": 6, "expected": {"kind": "sha256", "value": "b" * 64}}}}
    archive.api = SimpleNamespace(repo_info=lambda *a, **k: SimpleNamespace(siblings=[]))
    receipt = {"remote": remote, "commit": "a" * 40, "size": 6, "sha256": hashlib.sha256(b"abcdef").hexdigest()}
    with pytest.raises(job.FatalJobError, match="no cleanup"):
        archive.verify(receipt)


def test_upload_failure_retains_volume(tmp_path):
    target = tmp_path / "shard-00000.npy"
    target.write_bytes(b"abcdef")
    remote = "classical/tv/config/volumes/Test/shard-00000.npy"
    archive = archive_with(Session(*(Response(b"wrong") for _ in range(12))))
    archive.api = SimpleNamespace(
        dataset_info=lambda _: SimpleNamespace(private=True),
        create_commit=lambda **kwargs: SimpleNamespace(oid="a" * 40),
        repo_info=lambda *a, **k: SimpleNamespace(
            sha=k["revision"],
            siblings=[SimpleNamespace(rfilename=remote, size=6, lfs=SimpleNamespace(sha256="b" * 64), blob_id=None)],
        ),
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


def test_cli_defaults_and_all_seven_methods():
    args = job.parser().parse_args(["--execute"])
    assert args.methods == list(job.METHODS)
    assert job.parser().parse_args(["--execute", "--methods", "all7"]).methods == ["all7"]
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
def test_multishard_transaction_and_pending_resume(tmp_path, monkeypatch, fail_upload):
    assert job.SHARD_SIZE == 128
    monkeypatch.setattr(job, "SHARD_SIZE", 2)
    monkeypatch.setattr(job, "SPLITS", {"Training": (4, 2, 2)})
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
        if fail_upload and remote.endswith("shard-00001.npy"):
            raise job.FatalJobError("Upload failed")
        return {"remote": remote, "sha256": job.digest(path), "size": path.stat().st_size, "commit": "a" * 40}

    def upload_many(items, commit_message):
        return [upload(path, remote) for path, remote in items]

    archive = SimpleNamespace(
        upload=upload, upload_many=upload_many, verify=verified.append, remote_completed=lambda _: None
    )
    args = SimpleNamespace(work_dir=tmp_path, drive_dir=None, wait_gpu_seconds=0)
    arrays = {
        "Training": (
            np.broadcast_to(np.zeros((200, 200, 200), dtype=np.uint8), (4, 200, 200, 200)),
            np.array([0, 1, 0, 1]),
        )
    }
    suite = SimpleNamespace(MetricContext=Context, metric_protocol={"version": "test"}, release_memory=lambda: None)
    cp = SimpleNamespace(asnumpy=np.asarray)
    run = SimpleNamespace(log=lambda _: None, log_artifact=lambda _: None)
    baseline = tmp_path / "original" / "baseline"
    baseline.mkdir(parents=True)
    (baseline / "per_volume.csv").write_text(
        "split,index,class,SNR\nTraining,0,0,1.0\nTraining,1,1,1.0\nTraining,2,0,1.0\nTraining,3,1,1.0\n"
    )
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
        assert len(list(tmp_path.rglob("*.receipt.json"))) == 1
        assert len(list(tmp_path.rglob("*.pending.json"))) == 2
        assert len(calls) == 4
        fail_upload = False
    result = job.run_method(*parameters)
    assert result["status"] == "complete"
    assert job.read_json(next(tmp_path.rglob("_COMPLETE.json")))["volumes"] == 4
    assert len(calls) == 4
    for number, pending_path in enumerate(sorted(tmp_path.rglob("*.pending.json"))):
        identity = job.read_json(pending_path)["identity"]
        assert identity == {
            "dataset": job.DESTINATION,
            "prefix": result["prefix"],
            "config_id": result["config_id"],
            "source": {"repo": job.SOURCE, "revision": job.REVISION},
            "split": "Training",
            "start": 2 * number,
            "end": 2 * number + 2,
        }
    assert not list(tmp_path.rglob("*.npy"))
    assert len(list(tmp_path.rglob("*.slices.jsonl.gz"))) == 2
    import pandas as pd

    aggregate = pd.read_csv(next(tmp_path.rglob("aggregate.csv")))
    assert set(aggregate["scope"]) == {"split_class", "split", "whole_dataset"}
    paired = pd.read_csv(next(tmp_path.rglob("paired_vs_raw.csv")))
    assert paired["SNR_minus_raw"].eq(0).all()
    result = job.run_method(*parameters)
    assert result["resumed_remote"]
    assert len(calls) == 4 and len(verified) >= 9


def test_pending_identity_recovery_preserves_artifacts(tmp_path):
    target = tmp_path / "shard-00001.npy"
    target.write_bytes(b"data")
    receipt_path = tmp_path / "shard-00001.receipt.json"
    pending_path = tmp_path / "shard-00001.pending.json"
    identity = {
        "dataset": job.DESTINATION,
        "prefix": "classical/wavelet/test",
        "config_id": "test",
        "source": {"repo": job.SOURCE, "revision": job.REVISION},
        "split": "Training",
        "start": 2,
        "end": 4,
    }
    artifact = {
        "local": str(target.resolve()),
        "remote": "classical/wavelet/test/volumes/Training/shard-00001.npy",
        "size": 4,
        "sha256": job.digest(target),
        "git_sha1": job.digest(target, "sha1", git=True),
    }
    pending = {
        "identity": {"split": "Training", "index": 1, "class": 1, "start": 2, "end": 4},
        "split": "Training",
        "start": 2,
        "end": 4,
        "generated": str(target.resolve()),
        "artifacts": [artifact],
        "metadata": {"method": "wavelet"},
        "metric_protocol": {"version": "test"},
    }
    job.atomic_json(pending_path, pending)
    before = pending_path.read_bytes()
    uploads = []

    def upload(items, message):
        uploads.append(items)
        return [{**artifact, "commit": "a" * 40}]

    archive = SimpleNamespace(upload_many=upload)
    with pytest.raises(ValueError, match="identity mismatch; files preserved"):
        job.upload_pending(receipt_path, pending_path, archive, identity)
    assert not uploads and not receipt_path.exists()
    assert pending_path.read_bytes() == before and target.read_bytes() == b"data"
    assert job.digest(target) == artifact["sha256"]
    assert job.digest(target, "sha1", git=True) == artifact["git_sha1"]
    pending["identity"] = identity
    job.atomic_json(pending_path, pending)
    receipt = job.upload_pending(receipt_path, pending_path, archive, identity)
    assert receipt["verified"] and job.read_json(receipt_path) == receipt
    assert len(uploads) == 1 and target.read_bytes() == b"data"
    assert job.read_json(pending_path) == pending


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


def test_visibility_must_match_explicit_public_authorization(monkeypatch):
    import huggingface_hub

    api = SimpleNamespace(
        whoami=lambda: {"name": "owner"}, dataset_info=lambda _: object(), update_repo_settings=lambda *a, **k: None
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(job.Archive, "privacy", lambda self: True)
    with pytest.raises(RuntimeError, match="still private"):
        job.Archive("secret")


def test_public_repo_is_never_changed_back_to_private(monkeypatch):
    import huggingface_hub

    changes = []
    api = SimpleNamespace(
        whoami=lambda: {"name": "owner"},
        dataset_info=lambda _: object(),
        update_repo_settings=lambda *a, **k: changes.append(k),
    )
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda **kwargs: api)
    monkeypatch.setattr(job.Archive, "privacy", lambda self: False)
    job.Archive("secret").ensure_visibility(ttl=0)
    assert changes == []


def test_remote_completion_checks_every_shard():
    artifact = {
        "remote": "classical/tv/id/volumes/Test/shard-00000.npy",
        "commit": "a" * 40,
        "size": 4,
        "sha256": hashlib.sha256(b"data").hexdigest(),
    }
    manifest = {
        "complete": True,
        "shards": [
            {"artifacts": [artifact], "split": split, "start": 0, "end": count, "verified": True}
            for split, (count, _, _) in job.SPLITS.items()
        ],
    }
    body = json.dumps(manifest).encode()
    manifest_receipt = {
        "remote": "classical/tv/id/manifest.json",
        "commit": "a" * 40,
        "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "local": "C:/old-machine/manifest.json",
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


@pytest.mark.parametrize("sentinel", ["STOP", "BLOCKED", "BLOCKED.json"])
def test_startup_sentinel_precedes_auth(tmp_path, monkeypatch, sentinel):
    (tmp_path / sentinel).touch()
    monkeypatch.setattr(job, "Archive", lambda *_: pytest.fail("Must not authenticate"))
    with pytest.raises(SystemExit, match="sentinel present"):
        job.main(["--execute", "--methods", "all7", "--work-dir", str(tmp_path)])
    assert not (tmp_path / "job.lock").exists()


def test_metadata_cache_is_pinned_and_reused():
    archive = archive_with(Session())
    calls = []
    remote = "test.bin"

    def info(*args, **kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            sha=kwargs["revision"],
            siblings=[
                SimpleNamespace(rfilename=remote, size=6, lfs=SimpleNamespace(sha256=kwargs["revision"]), blob_id=None)
            ],
        )

    archive.api = SimpleNamespace(repo_info=info)
    for revision in ("a" * 40, "b" * 40, "a" * 40):
        archive.verify({"remote": remote, "commit": revision, "size": 6, "sha256": revision})
    assert [c["revision"] for c in calls] == ["a" * 40, "b" * 40]
    assert not archive.session.calls


def test_mutated_local_cannot_verify_native_git_receipt(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"before")
    receipt = {"remote": "report.json", "commit": "a" * 40, "size": 6, "sha256": job.digest(path), "local": str(path)}
    path.write_bytes(b"after!")
    archive = archive_with(Session(Response(b"before")))
    archive._index = {
        "a" * 40: {"report.json": {"size": 6, "expected": {"kind": "git", "value": job.digest(path, "sha1", git=True)}}}
    }
    archive.verify(receipt)
    assert len(archive.session.calls) == 1


def test_stored_git_hash_verifies_without_local_file():
    archive = archive_with(Session())
    archive._index = {"a" * 40: {"report.json": {"size": 6, "expected": {"kind": "git", "value": "b" * 40}}}}
    archive.verify({"remote": "report.json", "commit": "a" * 40, "size": 6, "sha256": "c" * 64, "git_sha1": "b" * 40})
    assert not archive.session.calls


def test_upload_retry_rebuilds_mutated_operations(tmp_path):
    path = tmp_path / "report.json"
    path.write_bytes(b"report")
    archive = archive_with(Session())
    calls = []

    def commit(**kwargs):
        ops = kwargs["operations"]
        calls.append(ops)
        if len(calls) == 1:
            ops[0]._is_uploaded = True
            raise OSError("Temporary upload error")
        assert ops[0] is not calls[0][0]
        assert not ops[0]._is_uploaded
        return SimpleNamespace(oid="a" * 40)

    archive.api = SimpleNamespace(create_commit=commit)
    archive.verify = lambda _: None
    receipt = archive.upload(path, "report.json")
    assert len(calls) == 2
    assert receipt["git_sha1"] == job.digest(path, "sha1", git=True)
    assert receipt["local"] == str(path.resolve())


def test_metadata_retry_respects_retry_after(monkeypatch):
    class Throttled(Exception):
        response = SimpleNamespace(headers={"Retry-After": "450"})

    sleeps, calls = [], []
    monkeypatch.setattr(job.time, "sleep", sleeps.append)

    def info(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise Throttled()
        return SimpleNamespace(sha=kwargs["revision"], siblings=[])

    archive = archive_with(Session())
    archive.api = SimpleNamespace(repo_info=info)
    assert archive.remote_index("a" * 40) == {}
    assert sleeps == [450]


@pytest.mark.parametrize("changed", ["bytes", "identity"])
def test_pending_shard_rejects_mutations(tmp_path, changed):
    path = tmp_path / "shard-00000.npy"
    path.write_bytes(b"data")
    pending_path = tmp_path / "pending.json"
    identity = {"dataset": job.DESTINATION, "config_id": "one"}
    job.atomic_json(
        pending_path,
        {
            "identity": identity,
            "artifacts": [{"local": str(path), "remote": "shard.npy", "sha256": job.digest(path), "size": 4}],
        },
    )
    if changed == "bytes":
        path.write_bytes(b"oops")
    else:
        identity = {**identity, "config_id": "two"}
    archive = SimpleNamespace(upload_many=lambda *_: pytest.fail("Must not upload invalid pending data"))
    with pytest.raises(ValueError, match="preserved"):
        job.upload_pending(tmp_path / "receipt.json", pending_path, archive, identity)
    assert path.exists() and pending_path.exists()


@pytest.mark.parametrize("name", ["wavelet", "nlm"])
@pytest.mark.parametrize("failure", [None, "parity", "exception"])
def test_extra_validation_full_and_diverse_bounded(tmp_path, monkeypatch, name, failure):
    inputs, checked = [], []

    class Training:
        def __getitem__(self, index):
            inputs.append(index)
            return np.full((200, 200, 200), index % 256, np.uint8)

    class Backend:
        metadata = {"implementation_sha256": "fixed"}

        def validate(self, raw):
            assert raw.shape == (200, 200, 200)
            if failure == "exception":
                raise RuntimeError("validation failed")
            return {"passed": True}

    backend = Backend()
    backend.name = name

    def planes(backend, volume, planes):
        checked.append((int(volume[0, 0, 0]), planes))
        return {"passed": failure != "parity", "planes": planes}

    monkeypatch.setattr(job, "validate_oct_planes", planes)
    path = tmp_path / "validation.json"
    if failure == "exception":
        with pytest.raises(RuntimeError):
            job.validate_extra_backend(backend, Training(), path)
    else:
        result = job.validate_extra_backend(backend, Training(), path)
        assert result["passed"] == (failure is None)
        assert checked == [(0, 200), (25, 8), (100, 8), (232, 8)]
        assert inputs == [0, 0, 25, 100, 1000]
    saved = job.read_json(path)
    assert saved["passed"] == (failure is None)
    assert saved["seconds"] >= 0
    assert saved["metadata"]["implementation_sha256"] == "fixed"


@pytest.mark.parametrize("name", ["wavelet", "nlm"])
def test_extra_independent_reference_rejects_passthrough(name):
    class Backend:
        cp = SimpleNamespace(asnumpy=np.asarray)

        def __call__(self, raw):
            return raw

    backend = Backend()
    backend.name = name
    volume = np.random.default_rng(12).integers(0, 256, (200, 200, 200), dtype=np.uint8)
    assert not job.validate_oct_planes(backend, volume, planes=1)["passed"]


@pytest.mark.parametrize("name,classname", [("wavelet", "WaveletBackend"), ("nlm", "NLMBackend")])
def test_extra_factory_only_imports_requested_module(monkeypatch, name, classname):
    import sys

    sentinel = object()
    monkeypatch.setitem(sys.modules, f"scripts.denoise_{name}_cuda", SimpleNamespace(**{classname: lambda: sentinel}))
    assert job.get_extra_backend(name) is sentinel


def test_incomplete_counts_never_pass_coverage():
    shards = [{"verified": True, "split": "Training", "start": 0, "end": 2}]
    with pytest.raises(ValueError, match="coverage"):
        job.verify_coverage(shards)


@pytest.mark.parametrize(
    "method,cid,parameters",
    [
        ("gaussian", "cbc335577ceaf03f8442", {"sigma": [0, 1.5, 1.5], "mode": "nearest", "truncate": 4.0}),
        ("median", "351aadadb7bc97665be3", {"size": [1, 3, 3], "mode": "nearest"}),
        ("tv", "901f1d6dcf69cae9d2e0", {"weight": 0.1, "eps": 0.0002, "max_num_iter": 200, "batch_planes": 32}),
    ],
)
def test_frozen_original_config_ids(method, cid, parameters):
    from scripts.denoise_cuda_suite import metric_protocol

    metadata = {
        "method": method,
        "parameters": parameters,
        "implementation": "scripts.denoise_cuda_suite.Backend",
        "implementation_sha256": "3c41fe80c7e5dc234e4e1b4a87c36155467b36cec55eafa4fa1093b0b5292166",
        "backend": "cupy",
        "cupy_version": "14.2.0",
        "quantization": "float32 /255, clip(filtered*255,0,255), uint8 truncation",
        "variant": "historical-2d",
    }
    if method == "gaussian":
        metadata.update(
            implementation="scripts.denoise_cuda_suite._CudaGaussian",
            implementation_sha256="3530b1c75d66cedd6921cbd89f9ab8b0908f35b84956b59b6e49ad2669e382c8",
            arithmetic="SciPy symmetric center-first outer-to-inner float64 sum; float32 per-axis output; no FMA",
            coefficients_sha256="b538f9793eff6803daf93387aec7c2ce4380e3cbc592d2c4521c85eb719c82e5",
            scipy_version="1.18.1",
        )
    assert job.config_id(metadata, {"repo": job.SOURCE, "revision": job.REVISION}, metric_protocol) == cid


@pytest.mark.parametrize("failed", [None, "nlm"])
def test_main_all7_reuses_completed_methods_and_reports_honestly(tmp_path, monkeypatch, failed):
    import sys

    import wandb
    from pipeline import utils
    from scripts import denoise_cuda_suite as suite

    monkeypatch.setattr(utils, "load_env_file", lambda: None)
    monkeypatch.setenv("HF_TOKEN", "test")
    monkeypatch.setenv("WANDB_API_KEY", "test")
    monkeypatch.setattr(job, "patch_cupy_gaussian", lambda: None)
    monkeypatch.setattr(job, "wait_gpu", lambda *_: None)
    cp = SimpleNamespace(cuda=SimpleNamespace(get_current_stream=lambda: SimpleNamespace(synchronize=lambda: None)))
    monkeypatch.setitem(sys.modules, "cupy", cp)
    uploaded, computed, checks, pilots = [], [], [], []
    archive = SimpleNamespace(upload=lambda path, remote: uploaded.append(remote) or {"remote": remote})
    monkeypatch.setattr(job, "Archive", lambda _: archive)
    run = SimpleNamespace(
        settings=SimpleNamespace(mode="online"), url="test", summary={}, log=lambda _: None, finish=lambda **_: None
    )
    monkeypatch.setattr(wandb, "init", lambda **_: run)
    training = [np.zeros((200, 200, 200), np.uint8)] * 12
    monkeypatch.setattr(job, "load_sources", lambda *_args, **_kwargs: ({"Training": (training, np.zeros(12))}, {}))
    monkeypatch.setattr(job, "load_bilateral", lambda *_: (None, {"method": "bilateral"}))

    class Backend:
        def __init__(self, name):
            self.name, self.metadata = name, {"method": name, "implementation_sha256": "test"}

        def __call__(self, raw):
            pilots.append(self.name)
            return raw

    monkeypatch.setattr(suite, "get_backend", lambda name, **_: Backend(name))
    monkeypatch.setattr(job, "get_extra_backend", Backend)
    monkeypatch.setattr(suite, "release_memory", lambda: None)
    monkeypatch.setattr(suite, "MetricContext", lambda _: SimpleNamespace(evaluate=lambda _: ({}, [])))

    def validate(backend, raw, path):
        checks.append(backend.name)
        return {"passed": backend.name != failed}

    monkeypatch.setattr(job, "validate_extra_backend", validate)

    def method(*args, resume_only=False):
        name = args[1]
        if resume_only and name in ("wavelet", "nlm"):
            return None
        if not resume_only:
            computed.append(name)
        return {"status": "complete", "config_id": name, "resumed_remote": resume_only}

    monkeypatch.setattr(job, "run_method", method)
    code = job.main(["--execute", "--methods", "all7", "--work-dir", str(tmp_path)])
    state = job.read_json(tmp_path / "status.json")
    assert code == (2 if failed else 0)
    assert state["all_seven_complete"] is (failed is None)
    assert state["selected_methods_complete"] is (failed is None)
    assert checks == ["wavelet", "nlm"]
    assert computed == (["wavelet"] if failed else ["wavelet", "nlm"])
    assert pilots.count("wavelet") == 12
    assert pilots.count("nlm") == (0 if failed else 12)
    assert job.read_json(tmp_path / "wavelet-pilot.json")["metadata"]["implementation_sha256"] == "test"
    assert job.read_json(tmp_path / "nlm-validation.json")["passed"] is (failed is None)
    assert "await CUDA parity" not in (tmp_path / "README.md").read_text()
    assert ("_COMPLETE_SELECTED.json" in uploaded) is (failed is None)
