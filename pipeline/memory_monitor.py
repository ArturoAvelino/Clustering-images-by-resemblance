"""Low-frequency process RAM reporting for pipeline runs."""

from __future__ import annotations

import threading

import psutil
from tqdm import tqdm


class PipelineMemoryMonitor:
    """Display pipeline and worker RSS against total system RAM.

    Memory is sampled immediately and then every ``interval_seconds``. Process
    and child-process memory is summed, so image-loader workers are included.
    The interval comes from ``PipelineConfig.memory_refresh_interval_seconds``.
    The monitor uses one daemon thread and does no work between samples.
    """

    def __init__(self, interval_seconds: float = 60.0) -> None:
        self.interval_seconds = interval_seconds
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._bar = None
        self._process = psutil.Process()
        self._total_memory = psutil.virtual_memory().total

    @staticmethod
    def _gib(value: int) -> str:
        return f"{value / (1024 ** 3):.1f} GiB"

    def _sample(self) -> int:
        processes = [self._process]
        try:
            processes.extend(self._process.children(recursive=True))
        except psutil.Error:
            pass
        total = 0
        for process in processes:
            try:
                total += process.memory_info().rss
            except psutil.Error:
                # A short-lived worker may exit between enumeration and query.
                continue
        return total

    def _update(self) -> None:
        if self._bar is None:
            return
        used = self._sample()
        self._bar.n = min(used, self._total_memory)
        self._bar.set_postfix_str(
            f"{self._gib(used)} / {self._gib(self._total_memory)} total RAM",
            refresh=False,
        )
        self._bar.refresh()

    def _run(self) -> None:
        while not self._stop.wait(self.interval_seconds):
            self._update()

    def __enter__(self) -> "PipelineMemoryMonitor":
        self._bar = tqdm(
            total=self._total_memory,
            desc="Pipeline RAM",
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            position=2,
            bar_format="{desc}: {bar} {postfix}",
        )
        self._update()
        self._thread = threading.Thread(
            target=self._run, name="pipeline-memory-monitor", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_seconds + 1.0))
        if self._bar is not None:
            self._bar.close()
