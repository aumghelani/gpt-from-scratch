import numpy as np
import pytest

from gpt_scratch.data.shards import load_shard, write_shards


def test_split_sizes_and_roundtrip(tmp_path):
    ids = list(range(1000))
    meta = write_shards(ids, tmp_path, vocab_size=1000, val_fraction=0.1)
    train = load_shard(tmp_path / "train.bin")
    val = load_shard(tmp_path / "val.bin")
    assert len(train) == 900 and len(val) == 100
    assert train.tolist() + val.tolist() == ids  # contiguous split, no shuffling
    assert meta["n_train"] == 900 and meta["n_val"] == 100


def test_shards_are_uint16_memmaps(tmp_path):
    write_shards([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], tmp_path, vocab_size=11)
    arr = load_shard(tmp_path / "train.bin")
    assert arr.dtype == np.uint16 and isinstance(arr, np.memmap)


def test_rejects_vocab_too_large_for_uint16(tmp_path):
    with pytest.raises(ValueError):
        write_shards([1, 2, 3], tmp_path, vocab_size=70_000)


def test_rejects_ids_outside_vocab(tmp_path):
    with pytest.raises(ValueError):
        write_shards([0, 1, 99], tmp_path, vocab_size=10)


def test_rejects_too_little_data(tmp_path):
    with pytest.raises(ValueError):
        write_shards([1, 2], tmp_path, vocab_size=10, val_fraction=0.5)


def test_meta_file_written(tmp_path):
    write_shards(list(range(100)), tmp_path, vocab_size=100)
    assert (tmp_path / "meta.json").exists()
