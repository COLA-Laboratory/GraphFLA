"""Bounded-runner failures terminate their subprocesses instead of hanging."""

import os
from pathlib import Path
import subprocess
import sys

import pytest

from tools._resource_guard import ResourceLimitError, run_guarded

pytestmark = pytest.mark.skipif(os.name != "posix", reason="POSIX benchmark runner")


def test_guard_returns_output_and_checks_exit_status():
    result = run_guarded(
        [sys.executable, "-c", "print('done')"], timeout=5, memory_bytes=128 * 1024**2
    )
    assert result.stdout.strip() == "done" and result.returncode == 0
    with pytest.raises(subprocess.CalledProcessError) as error:
        run_guarded(
            [sys.executable, "-c", "import sys; sys.exit(7)"],
            timeout=5,
            memory_bytes=128 * 1024**2,
        )
    assert error.value.returncode == 7


def test_guard_enforces_wall_timeout():
    with pytest.raises(ResourceLimitError, match="Wall time"):
        run_guarded(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            timeout=0.15,
            memory_bytes=128 * 1024**2,
        )


def test_guard_counts_child_memory_and_terminates_child(tmp_path):
    import psutil

    pidfile = tmp_path / "child.pid"
    child = "import time; x=bytearray(64*1024**2); time.sleep(10)"
    parent = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child!r}]); "
        f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(10)"
    )
    with pytest.raises(ResourceLimitError, match="RSS"):
        run_guarded(
            [sys.executable, "-c", parent], timeout=5, memory_bytes=48 * 1024**2
        )
    pid = int(pidfile.read_text())
    # A killed orphan may briefly remain a zombie until the OS reaps it.
    try:
        process = psutil.Process(pid)
        process.wait(timeout=2)
    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
        assert (
            not psutil.pid_exists(pid)
            or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
        )


def test_worker_rejects_unregistered_expanding_case():
    repo = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            sys.executable,
            str(repo / "tools/benchmark_analysis.py"),
            "--worker",
            "--metric",
            "global_idiosyncratic_index",
            "--case",
            "boolean-100",
        ],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "not registered" in result.stderr
