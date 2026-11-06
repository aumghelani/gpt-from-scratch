import hashlib

import pytest

from gpt_scratch.data.download import ChecksumError, download_file, sha256_of


@pytest.fixture
def source(tmp_path):
    f = tmp_path / "src.txt"
    f.write_bytes(b"to be or not to be")
    return f


def test_downloads_and_verifies(tmp_path, source):
    digest = hashlib.sha256(b"to be or not to be").hexdigest()
    dest = tmp_path / "out" / "data.txt"
    result = download_file(source.as_uri(), dest, expected_sha256=digest)
    assert result == dest and dest.read_bytes() == b"to be or not to be"


def test_checksum_mismatch_raises_and_leaves_no_file(tmp_path, source):
    dest = tmp_path / "data.txt"
    with pytest.raises(ChecksumError):
        download_file(source.as_uri(), dest, expected_sha256="0" * 64)
    assert not dest.exists()
    assert not list(tmp_path.glob("*.part"))


def test_skips_when_valid_file_already_present(tmp_path, source):
    digest = sha256_of(source)
    dest = tmp_path / "data.txt"
    dest.write_bytes(source.read_bytes())
    missing = (tmp_path / "gone.txt").as_uri()  # would fail if a fetch were attempted
    assert download_file(missing, dest, expected_sha256=digest) == dest


def test_redownloads_when_existing_file_is_corrupt(tmp_path, source):
    digest = sha256_of(source)
    dest = tmp_path / "data.txt"
    dest.write_bytes(b"corrupt")
    download_file(source.as_uri(), dest, expected_sha256=digest)
    assert dest.read_bytes() == source.read_bytes()
