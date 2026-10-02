"""Time/RSS watchdog for trusted local benchmark subprocesses (POSIX)."""

import os
import signal
import subprocess
import time

import psutil


class ResourceLimitError(RuntimeError):
    pass


def run_guarded(cmd, *, timeout, memory_bytes, **kwargs):
    """Run an isolated process group, polling its total RSS every 50 ms.

    RSS includes child processes; shared pages can be counted more than once.
    This is a sampled watchdog, not an OS reservation or an allocation limit.
    """
    if os.name != "posix":
        raise RuntimeError("The bounded runner requires POSIX; use ASV on Windows.")
    if timeout <= 0 or memory_bytes <= 0:
        raise ValueError("Resource limits must be positive.")
    started, peak = time.monotonic(), 0
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
        **kwargs,
    )
    try:
        monitor = psutil.Process(process.pid)
        while process.poll() is None:
            remaining = timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise ResourceLimitError(f"Wall time exceeded {timeout:g} seconds")
            try:
                family = [monitor, *monitor.children(recursive=True)]
                rss = 0
                for member in family:
                    try:
                        rss += member.memory_info().rss
                    except psutil.NoSuchProcess:
                        pass
                peak = max(peak, rss)
                if rss > memory_bytes:
                    raise ResourceLimitError(
                        f"Process-tree RSS {rss} exceeded {memory_bytes} bytes"
                    )
            except psutil.NoSuchProcess:
                break
            try:
                stdout, stderr = process.communicate(timeout=min(0.05, remaining))
                break
            except subprocess.TimeoutExpired:
                pass
        else:
            stdout, stderr = process.communicate()
        # Also drain pipes when the monitored process exits during a poll.
        stdout, stderr = process.communicate()
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode, cmd, stdout, stderr)
        result = subprocess.CompletedProcess(cmd, process.returncode, stdout, stderr)
        result.monitored_peak_rss_bytes = peak
        return result
    finally:
        # Include descendants that might outlive the leader, also on interruption.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.communicate()
