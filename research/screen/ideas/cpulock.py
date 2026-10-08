"""One heavy CPU job at a time across all agents of the ideas round (the laptop overheats otherwise).

  from cpulock import cpu_lock
  with cpu_lock("my_name"):
      H.run_all(...); H.evaluate_all(...)
Blocks until no other agent holds the lock. Hold it only around harness runs, embedding or model fitting,
never around LLM calls (those are network bound and may run in parallel).
"""
import sys as _zr_sys, pathlib as _zr_pl
_zr_sys.path.insert(0, str(next(p for p in _zr_pl.Path(__file__).resolve().parents if (p / "zr_paths.py").exists())))
import zr_paths as ZR  # noqa: E402
import fcntl
import os
import time
from contextlib import contextmanager

LOCK = ZR.SCREEN + "/ideas/.cpu.lock"


@contextmanager
def cpu_lock(who="?"):
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    f = open(LOCK, "a+")
    t0 = time.time()
    fcntl.flock(f, fcntl.LOCK_EX)
    with open(LOCK + ".log", "a") as lg:
        lg.write(f"{time.strftime('%H:%M:%S')} {who} acquired after {time.time() - t0:.0f}s\n")
    try:
        yield
    finally:
        with open(LOCK + ".log", "a") as lg:
            lg.write(f"{time.strftime('%H:%M:%S')} {who} released\n")
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()
