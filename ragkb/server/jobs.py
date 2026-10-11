"""Ingestion workers in two lanes, so a small file never waits behind a large one.

Jobs for files under ``SMALL_FILE_BYTES`` (1 MB) go to the small lane, the rest to the large lane.
With two or more workers (``RAGKB_INGEST_WORKERS``, default 2) the first worker takes only small
jobs and the others take small jobs first, then large ones; with one worker, small jobs still go
first but wait for a large job that is already running.

A job stays ``queued`` until a worker takes it; the runner marks it ``processing`` from there. The
worker threads start with the first job, so a process that never ingests starts none.
"""
from __future__ import annotations

import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("ragkb.server")

SMALL_FILE_BYTES = 1024 * 1024


@dataclass
class QueuedJob:
    job_id: str
    kind: str  # upload, reindex or retry
    collection_id: str
    size: int
    doc_id: str = ""
    filename: str = ""
    title: str | None = None
    source_path: str | None = None
    # an upload's bytes wait on disk, not in memory, until a worker reads them
    spool: Path | None = None
    # a retry: the document's error before it was queued, restored if the job never runs
    previous_error: str | None = None

    @property
    def small(self) -> bool:
        return self.size < SMALL_FILE_BYTES


class SchedulerStopped(RuntimeError):
    """The workers have stopped (the server is shutting down)."""


class IngestScheduler:
    def __init__(self, workers: int, runner: Callable[[QueuedJob], None], *, name: str = "ragkb-ingest") -> None:
        self.workers = max(1, int(workers))
        self._runner = runner
        self._name = name
        self._small: deque[QueuedJob] = deque()
        self._large: deque[QueuedJob] = deque()
        self._running: dict[str, QueuedJob] = {}
        self._cond = threading.Condition()
        self._threads: list[threading.Thread] = []
        self._stopped = False

    # ------------------------------------------------------------------ submitting and cancelling

    def submit(self, item: QueuedJob) -> None:
        with self._cond:
            if self._stopped:
                raise SchedulerStopped("The ingestion workers have stopped.")
            (self._small if item.small else self._large).append(item)
            if not self._threads:
                for index in range(self.workers):
                    thread = threading.Thread(target=self._work, args=(index,), name=f"{self._name}-{index}",
                                              daemon=True)
                    self._threads.append(thread)
                    thread.start()
            self._cond.notify_all()

    def cancel(self, job_id: str) -> tuple[str, QueuedJob | None]:
        """Take a job out of the queue if it is still waiting.

        Returns ``("queued", job)`` when it was removed, ``("running", job)`` when a worker has it,
        and ``("idle", None)`` when this process does not know it (finished, or left by a server
        that stopped).
        """
        with self._cond:
            for lane in (self._small, self._large):
                for item in lane:
                    if item.job_id == job_id:
                        lane.remove(item)
                        return "queued", item
            running = self._running.get(job_id)
            return ("running", running) if running is not None else ("idle", None)

    def state(self, job_id: str) -> str:
        """``queued``, ``running`` or ``idle`` (not known here)."""
        with self._cond:
            if job_id in self._running:
                return "running"
            if any(item.job_id == job_id for item in (*self._small, *self._large)):
                return "queued"
            return "idle"

    def active(self, *, doc_id: str | None = None, collection_id: str | None = None) -> list[str]:
        """Ids of the waiting and running jobs for one document or one collection."""
        with self._cond:
            items = [*self._running.values(), *self._small, *self._large]
        return [
            item.job_id
            for item in items
            if (doc_id is None or item.doc_id == doc_id)
            and (collection_id is None or item.collection_id == collection_id)
        ]

    def counts(self) -> dict[str, int]:
        with self._cond:
            return {"queued": len(self._small) + len(self._large), "running": len(self._running)}

    def shutdown(self) -> list[QueuedJob]:
        """Stop taking jobs and return the ones that never started. Running jobs finish on their threads."""
        with self._cond:
            self._stopped = True
            left = [*self._small, *self._large]
            self._small.clear()
            self._large.clear()
            self._cond.notify_all()
        return left

    # ------------------------------------------------------------------ workers

    def _take(self, small_only: bool) -> QueuedJob | None:
        if self._small:
            return self._small.popleft()
        if not small_only and self._large:
            return self._large.popleft()
        return None

    def _work(self, index: int) -> None:
        small_only = self.workers > 1 and index == 0
        while True:
            with self._cond:
                item = None if self._stopped else self._take(small_only)
                while item is None and not self._stopped:
                    self._cond.wait()
                    item = None if self._stopped else self._take(small_only)
                if item is None:
                    return
                self._running[item.job_id] = item
            try:
                self._runner(item)
            except Exception:
                log.exception("an ingestion job stopped unexpectedly", extra={"job_id": item.job_id})
            finally:
                with self._cond:
                    self._running.pop(item.job_id, None)
