import pyLOCO.parallel as parallel


def test_worker_count_respects_slurm_allocation(monkeypatch):
    monkeypatch.setattr(parallel.multiprocessing, "cpu_count", lambda: 40)
    monkeypatch.setattr(
        parallel.os, "sched_getaffinity", lambda _pid: set(range(32)), raising=False
    )
    assert parallel.available_worker_count(
        task_count=1350,
        environ={"SLURM_CPUS_PER_TASK": "16"},
    ) == 16


def test_worker_count_uses_smallest_explicit_and_environment_limit(monkeypatch):
    monkeypatch.setattr(parallel.multiprocessing, "cpu_count", lambda: 40)
    monkeypatch.setattr(
        parallel.os, "sched_getaffinity", lambda _pid: set(range(24)), raising=False
    )
    assert parallel.available_worker_count(
        requested=8,
        task_count=3,
        environ={"SLURM_CPUS_PER_TASK": "16"},
    ) == 3


def test_invalid_slurm_value_falls_back_to_visible_cpus(monkeypatch):
    monkeypatch.setattr(parallel.multiprocessing, "cpu_count", lambda: 40)
    monkeypatch.setattr(
        parallel.os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False
    )
    assert parallel.available_worker_count(
        task_count=20,
        environ={"SLURM_CPUS_PER_TASK": "unknown"},
    ) == 12
