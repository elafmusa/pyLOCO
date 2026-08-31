"""Helpers for respecting scheduler and process-affinity CPU allocations."""

from __future__ import annotations

import multiprocessing
import os
import re


def available_worker_count(*, requested=None, task_count=None, environ=None):
    """Return a positive worker count bounded by the current allocation.

    ``multiprocessing.cpu_count()`` reports host CPUs on systems where a Slurm
    job is not constrained by a CPU-visible cgroup.  Prefer the smallest of
    the host count, process affinity, and ``SLURM_CPUS_PER_TASK``.  Explicit
    requests and the number of independent tasks are bounds as well.
    """
    env = os.environ if environ is None else environ
    limits = [int(multiprocessing.cpu_count())]

    get_affinity = getattr(os, "sched_getaffinity", None)
    if get_affinity is not None:
        try:
            affinity_count = len(get_affinity(0))
        except (OSError, TypeError):
            affinity_count = 0
        if affinity_count > 0:
            limits.append(int(affinity_count))

    slurm_value = str(env.get("SLURM_CPUS_PER_TASK", "")).strip()
    match = re.match(r"^(\d+)", slurm_value)
    if match and int(match.group(1)) > 0:
        limits.append(int(match.group(1)))

    for value in (requested, task_count):
        if value is not None:
            value = int(value)
            if value <= 0:
                raise ValueError("Worker-count limits must be positive")
            limits.append(value)

    return max(1, min(limits))
