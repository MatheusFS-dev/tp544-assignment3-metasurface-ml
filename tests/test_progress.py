"""Console progress contracts for durable stage reporting."""

import pytest

import metasurface.common.progress as progress


def test_progress_reports_resume_unknown_eta_and_elapsed(monkeypatch, capsys):
    times = iter([10.0, 14.0, 18.0])
    monkeypatch.setattr(progress.time, "monotonic", lambda: next(times))
    reporter = progress.Progress("Stage", 10, saved=3)
    reporter.committed(3)
    reporter.committed(5)
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "Stage: starting with 3/10 saved, ETA unknown"
    assert "3/10 saved, elapsed 4s, ETA unknown" in lines[1]
    assert "5/10 saved, elapsed 8s, ETA 20s" in lines[2]


def test_progress_not_reported_when_commit_fails(monkeypatch, capsys):
    monkeypatch.setattr(progress.time, "monotonic", lambda: 0.0)
    reporter = progress.Progress("Stage", 1)

    def failed_save():
        raise OSError("disk full")

    with pytest.raises(OSError):
        failed_save()
        reporter.committed(1)
    assert "1/1 saved" not in capsys.readouterr().out
