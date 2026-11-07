import csv

from gpt_scratch.train.logger import CSVLogger


def read(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def test_writes_header_and_rows(tmp_path):
    with CSVLogger(tmp_path / "log.csv", fields=("step", "loss")) as log:
        log.log(step=1, loss=2.5)
        log.log(step=2, loss=2.0)
    rows = read(tmp_path / "log.csv")
    assert rows == [{"step": "1", "loss": "2.5"}, {"step": "2", "loss": "2.0"}]


def test_missing_fields_become_empty(tmp_path):
    with CSVLogger(tmp_path / "log.csv", fields=("step", "loss", "val_loss")) as log:
        log.log(step=1, loss=1.0)
    assert read(tmp_path / "log.csv")[0]["val_loss"] == ""


def test_append_mode_does_not_duplicate_header(tmp_path):
    p = tmp_path / "log.csv"
    with CSVLogger(p, fields=("step",)) as log:
        log.log(step=1)
    with CSVLogger(p, fields=("step",)) as log:
        log.log(step=2)
    assert [r["step"] for r in read(p)] == ["1", "2"]


def test_rows_flushed_immediately(tmp_path):
    p = tmp_path / "log.csv"
    log = CSVLogger(p, fields=("step",))
    log.log(step=1)
    assert len(read(p)) == 1  # visible before close
    log.close()


def test_unknown_field_rejected(tmp_path):
    import pytest

    with CSVLogger(tmp_path / "l.csv", fields=("step",)) as log, pytest.raises(KeyError):
        log.log(step=1, bogus=2)
