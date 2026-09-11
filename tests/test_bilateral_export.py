import hashlib
import io
import json
from unittest.mock import MagicMock, Mock, call

import numpy as np
import pytest
from skimage.restoration import denoise_bilateral

from scripts import prepare_bilateral_200 as export


@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("shape", [(2, 200, 200, 200), (2, 1, 200, 200, 200)])
def test_parse_volume_header_valid(shape, version):
    stream = io.BytesIO()
    writer = getattr(np.lib.format, f"write_array_header_{version}_0")
    writer(stream, {"descr": "|u1", "fortran_order": False, "shape": shape})
    payload = stream.getvalue()
    assert export.parse_volume_header(payload, len(payload) + int(np.prod(shape))) == {
        "shape": list(shape),
        "dtype": "uint8",
        "offset": len(payload),
        "sample_bytes": 200**3,
    }


@pytest.mark.parametrize(
    "shape,dtype,fortran,size_delta,message",
    [
        ((2, 199, 200, 200), "u1", False, 0, "Raw resolution"),
        ((2, 2, 200, 200, 200), "u1", False, 0, "Raw resolution"),
        ((0, 200, 200, 200), "u1", False, 0, "Raw resolution"),
        ((200, 200, 200), "u1", False, 0, "4D/5D"),
        ((2, 1, 1, 200, 200, 200), "u1", False, 0, "4D/5D"),
        ((2, 200, 200, 200), "f4", False, 0, "uint8"),
        ((2, 200, 200, 200), "u1", True, 0, "C-order"),
        ((2, 200, 200, 200), "u1", False, -1, "file size"),
        ((2, 200, 200, 200), "u1", False, 1, "file size"),
    ],
)
def test_parse_volume_header_rejects_invalid(shape, dtype, fortran, size_delta, message):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(
        stream, {"descr": np.dtype(dtype).str, "fortran_order": fortran, "shape": shape}
    )
    payload = stream.getvalue()
    file_size = len(payload) + int(np.prod(shape)) * np.dtype(dtype).itemsize + size_delta
    with pytest.raises(ValueError, match=message):
        export.parse_volume_header(payload, file_size)


@pytest.fixture
def http(monkeypatch):
    sleep = Mock()
    monkeypatch.setattr(export.time, "sleep", sleep)
    response = MagicMock()
    response.__enter__.return_value = response
    session = Mock()
    session.get.return_value = response
    return session, response, sleep


@pytest.mark.parametrize("status,start,end,size", [(206, 2, 5, 10), (200, 0, 3, 4)])
def test_read_range_accepts_exact_response(http, status, start, end, size):
    session, response, sleep = http
    response.status_code = status
    response.headers = {"Content-Range": f"bytes {start}-{end}/{size}"} if status == 206 else {}
    response.raw.read.return_value = b"abcd"
    headers = {"Authorization": "Bearer test-only"}
    assert export.read_range(session, "https://example.invalid/data.npy", start, end, size, headers) == b"abcd"
    session.get.assert_called_once_with(
        "https://example.invalid/data.npy",
        headers={**headers, "Range": f"bytes={start}-{end}", "Accept-Encoding": "identity"},
        stream=True,
        timeout=(30, 180),
    )
    response.raw.read.assert_called_once_with(5, decode_content=True)
    response.__exit__.assert_called_once()
    sleep.assert_not_called()
    assert headers == {"Authorization": "Bearer test-only"}


@pytest.mark.parametrize(
    "status,content_range,payload",
    [
        (200, "bytes 2-5/10", b"abcd"),
        (416, "bytes 2-5/10", b"abcd"),
        (206, None, b"abcd"),
        (206, "bytes 1-5/10", b"abcd"),
        (206, "bytes 2-6/10", b"abcd"),
        (206, "bytes 2-5/11", b"abcd"),
        (206, "bytes 2-5/*", b"abcd"),
        (206, "bytes 2-5/10", b"abc"),
        (206, "bytes 2-5/10", b"abcde"),
    ],
)
def test_read_range_rejects_invalid_response_without_sleeping(http, status, content_range, payload):
    session, response, sleep = http
    response.status_code = status
    response.headers = {} if content_range is None else {"Content-Range": content_range}
    response.raw.read.return_value = payload
    with pytest.raises(RuntimeError, match="after 6 attempts.*ValueError"):
        export.read_range(session, "https://example.invalid/data.npy", 2, 5, 10)
    assert session.get.call_count == 6
    assert response.__exit__.call_count == 6
    assert sleep.call_args_list == [call(1), call(2), call(4), call(8), call(16)]
    if status == 206 and content_range == "bytes 2-5/10":
        assert response.raw.read.call_args_list == [call(5, decode_content=True)] * 6
    else:
        response.raw.read.assert_not_called()


def test_read_range_retries_transient_failure(http):
    session, response, sleep = http
    session.get.side_effect = [OSError("connection interrupted"), response]
    response.status_code = 206
    response.headers = {"Content-Range": "bytes 2-5/10"}
    response.raw.read.return_value = b"abcd"
    assert export.read_range(session, "https://example.invalid/data.npy", 2, 5, 10) == b"abcd"
    assert session.get.call_count == 2
    sleep.assert_called_once_with(1)


def test_bilateral_matches_direct_skimage():
    volume = np.random.default_rng(42).integers(0, 256, size=(3, 4, 4), dtype=np.uint8)
    original = volume.copy()
    expected = np.empty_like(volume)
    for index, plane in enumerate(volume):
        image = plane.astype(np.float32) / 255.0
        filtered = denoise_bilateral(
            image,
            sigma_color=0.10,
            sigma_spatial=4.0,
            bins=10000,
            mode="constant",
            cval=0,
            channel_axis=None,
            win_size=None,
        )
        expected[index] = np.clip(np.nan_to_num(filtered, nan=image) * 255.0, 0, 255).astype(np.uint8)
    actual = export.bilateral(volume)
    assert actual.shape == volume.shape
    assert actual.dtype == np.uint8
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(volume, original)


@pytest.fixture
def writer_factory(tmp_path):
    writers = []

    def create(identity=None):
        writer = export.SplitWriter(
            tmp_path, "Training", {"revision": "test-revision"} if identity is None else identity, (3, 1, 4, 4, 4)
        )
        writers.append(writer)
        return writer

    yield create
    for writer in writers:
        writer.close()


def test_split_writer_out_of_order_resume_publish_and_hash(writer_factory):
    expected = np.arange(3 * 4**3, dtype=np.uint8).reshape(3, 1, 4, 4, 4)
    writer = writer_factory()
    assert writer.done == set()
    assert not writer.complete
    for index in (2, 0):
        writer.write(index, expected[index, 0])
    assert json.loads(writer.progress.read_text())["done"] == [0, 2]
    with pytest.raises(ValueError, match="incomplete"):
        writer.publish()
    assert not writer.target.exists()
    assert not writer.marker.exists()
    writer.close()

    resumed = writer_factory()
    assert resumed.done == {0, 2}
    np.testing.assert_array_equal(resumed.output[[0, 2]], expected[[0, 2]])
    with pytest.raises(ValueError, match="already committed"):
        resumed.write(2, expected[1, 0])
    resumed.write(1, expected[1, 0])
    assert not resumed.target.exists()
    assert not resumed.marker.exists()
    resumed.publish()
    assert resumed.complete
    assert resumed.output is None
    assert not resumed.partial.exists()
    np.testing.assert_array_equal(np.load(resumed.target, allow_pickle=False), expected)
    marker = json.loads(resumed.marker.read_text())
    assert marker == {
        "identity": resumed.identity,
        "shape": list(expected.shape),
        "dtype": "uint8",
        "file": resumed.target.name,
        "sha256": hashlib.sha256(resumed.target.read_bytes()).hexdigest(),
        "complete": True,
    }
    reopened = writer_factory()
    assert reopened.complete
    assert reopened.done == {0, 1, 2}
    assert reopened.output is None
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        writer_factory({"revision": "different"})
    corrupted = expected.copy()
    corrupted.flat[0] ^= 1
    np.save(reopened.target, corrupted)
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        writer_factory()


@pytest.mark.parametrize("suffix", ["npy", "partial.npy"])
def test_split_writer_does_not_overwrite_unverified(tmp_path, writer_factory, suffix):
    path = tmp_path / f"Training_volumes_dn.{suffix}"
    np.save(path, np.full((3, 1, 4, 4, 4), 17, dtype=np.uint8))
    original = path.read_bytes()
    with pytest.raises(FileExistsError, match="Unverified output"):
        writer_factory()
    assert path.read_bytes() == original
    assert set(tmp_path.iterdir()) == {path}


def test_split_writer_rejects_partial_identity_mismatch(writer_factory):
    writer = writer_factory()
    writer.write(2, np.full((4, 4, 4), 29, dtype=np.uint8))
    writer.close()
    original = writer.partial.read_bytes(), writer.progress.read_bytes()
    with pytest.raises(ValueError, match="identity mismatch"):
        writer_factory({"revision": "different"})
    assert (writer.partial.read_bytes(), writer.progress.read_bytes()) == original


def test_split_writer_recovers_crash_after_rename(writer_factory, monkeypatch):
    writer = writer_factory()
    expected = np.arange(3 * 4**3, dtype=np.uint8).reshape(3, 1, 4, 4, 4)
    for index in (2, 0, 1):
        writer.write(index, expected[index, 0])
    with monkeypatch.context() as patch:
        patch.setattr(export, "atomic_json", Mock(side_effect=OSError("simulated marker write failure")))
        with pytest.raises(OSError, match="simulated marker"):
            writer.publish()
    assert writer.target.exists()
    assert not writer.partial.exists()
    assert not writer.marker.exists()
    before = writer.target.read_bytes()
    recovered = writer_factory()
    assert recovered.complete
    assert recovered.done == {0, 1, 2}
    assert recovered.target.read_bytes() == before
    np.testing.assert_array_equal(np.load(recovered.target, allow_pickle=False), expected)
    assert json.loads(recovered.marker.read_text())["sha256"] == hashlib.sha256(before).hexdigest()
    assert writer_factory().complete


@pytest.mark.parametrize("stage", ["init", "publish"])
def test_split_writer_rejects_partial_and_unverified_final(writer_factory, stage):
    writer = writer_factory()
    for index in range(3):
        writer.write(index, np.full((4, 4, 4), index, dtype=np.uint8))
    if stage == "init":
        writer.close()
    np.save(writer.target, np.full(writer.shape, 99, dtype=np.uint8))
    paths = (writer.partial, writer.target, writer.progress)
    before = [path.read_bytes() for path in paths]

    with pytest.raises(FileExistsError, match="unverified final output"):
        if stage == "init":
            writer_factory()
        else:
            writer.publish()

    assert [path.read_bytes() for path in paths] == before
    assert not writer.marker.exists()
    assert not writer.complete


@pytest.mark.parametrize("mismatch", ["requested_shape", "marker_shape", "marker_dtype", "array_shape", "array_dtype"])
def test_split_writer_rejects_completed_shape_dtype_mismatch(writer_factory, mismatch):
    writer = writer_factory()
    for index in range(3):
        writer.write(index, np.full((4, 4, 4), index, dtype=np.uint8))
    writer.publish()
    marker = json.loads(writer.marker.read_text())
    requested_shape = writer.shape
    message = "Completed output shape/dtype mismatch"

    if mismatch == "requested_shape":
        requested_shape = (2, 1, 4, 4, 4)
    elif mismatch == "marker_shape":
        marker["shape"] = [2, 1, 4, 4, 4]
    elif mismatch == "marker_dtype":
        marker["dtype"] = "float32"
    else:
        array = np.load(writer.target, allow_pickle=False)
        array = array[:2] if mismatch == "array_shape" else array.astype(np.float32)
        np.save(writer.target, array)
        marker["sha256"] = hashlib.sha256(writer.target.read_bytes()).hexdigest()
        message = "Completed array shape/dtype mismatch"
    writer.marker.write_text(json.dumps(marker), encoding="utf-8")
    paths = (writer.target, writer.marker, writer.progress)
    before = [path.read_bytes() for path in paths]

    with pytest.raises(ValueError, match=message):
        reopened = export.SplitWriter(writer.directory, writer.split, writer.identity, requested_shape)
        reopened.close()

    assert [path.read_bytes() for path in paths] == before
    assert not writer.partial.exists()


def test_uploader_includes_completion_markers():
    from scripts import upload_bilateral_hf

    required = {"*_volumes_dn.npy", "*_labels.npy", "*_complete.json", "manifest.json", "README.md"}
    assert required <= set(upload_bilateral_hf.ALLOW)
