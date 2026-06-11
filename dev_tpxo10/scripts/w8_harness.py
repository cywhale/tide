#!/usr/bin/env python
"""§7.5.3 W8 harness driver (run in the dev env; servers run in the
PRODUCTION env via `uv run --project <repo>`).

Per open-mode {direct, dask (threaded auto), dist-auto, dist-native}:
  * launch production-like gunicorn: 2 uvicorn workers, NO --reload
  * dist-* modes: launch a dask scheduler + one worker --memory-limit 8GB
    (production topology, conf/simu.sh)
  * workloads: P-point x40 (latency), W6 45° sample=5 + W6b phase-3
    (overview5 confirmation), W8 45° sample=1 single x3 and
    concurrency-2 x2 (distinct-worker-PID asserted)
  * 50 ms process-tree RSS sampler -> per-Gunicorn-worker peak, dask
    worker peak, aggregate tree peak (G1-signed thresholds; baseline =
    post-warmup idle)

Signed thresholds (G1 kickoff, spec §7.5.3): Gunicorn worker delta<=1.5
GiB peak<=2 GiB; Dask worker delta<=2 GiB peak<=4 GiB; aggregate tree
delta<=3 GiB and host headroom>=25% at peak; payload<=100 MiB; wall<=15 s
single / <=30 s concurrency-2. STOP at G1 if any threshold fails.

Results: printed verdicts + JSON at benchmarks/w8_harness.json.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import psutil

REPO = Path(__file__).resolve().parents[2]
DEV = REPO / "dev_tpxo10"
PORT = 8841
SCHED_PORT = 8842
BASE = f"http://127.0.0.1:{PORT}"
GiB = 2**30
MODES = ["direct", "dask", "dist-auto", "dist-native"]
BENCH = dict(lon0=105.0, lon1=150.0, lat0=0.0, lat1=45.0)

THRESH = {
    "gunicorn_delta": 1.5 * GiB, "gunicorn_peak": 2 * GiB,
    "dask_delta": 2 * GiB, "dask_peak": 4 * GiB,
    "tree_delta": 3 * GiB, "headroom_frac": 0.25,
    "payload": 100 * 2**20, "wall_single": 15.0, "wall_conc2": 30.0,
}


class TreeSampler(threading.Thread):
    """50 ms peak-RSS sampler over named process groups."""

    def __init__(self, groups: dict[str, list[int]]):
        super().__init__(daemon=True)
        self.groups = {k: [psutil.Process(p) for p in v] for k, v in groups.items()}
        self.peak = {k: {p.pid: 0 for p in v} for k, v in self.groups.items()}
        self.tree_peak = 0
        self.min_avail = psutil.virtual_memory().available
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            total = 0
            for k, procs in self.groups.items():
                for p in procs:
                    try:
                        rss = p.memory_info().rss
                    except psutil.Error:
                        continue
                    self.peak[k][p.pid] = max(self.peak[k][p.pid], rss)
                    total += rss
            self.tree_peak = max(self.tree_peak, total)
            self.min_avail = min(self.min_avail,
                                 psutil.virtual_memory().available)
            time.sleep(0.05)

    def snapshot(self):
        return {k: dict(v) for k, v in self.peak.items()}

    def stop(self):
        self._halt.set()


def wait_port(port, timeout=90):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def get(url):
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=180) as r:
        body = r.read()
        pid = r.headers.get("X-Worker-PID")
    return time.perf_counter() - t0, len(body), pid


def find_workers(master_pid):
    master = psutil.Process(master_pid)
    for _ in range(60):
        kids = [c for c in master.children(recursive=True)]
        if len(kids) >= 2:
            return [c.pid for c in kids]
        time.sleep(0.5)
    raise RuntimeError("gunicorn workers not found")


def run_mode(mode: str, results: dict):
    env = os.environ.copy()
    env["TIDE_BENCH_OPEN"] = mode
    procs = []
    dask_pids = []
    try:
        if mode.startswith("dist"):
            sched = subprocess.Popen(
                ["uv", "run", "--project", str(REPO), "dask", "scheduler",
                 "--port", str(SCHED_PORT), "--no-dashboard"],
                cwd=DEV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(sched)
            if not wait_port(SCHED_PORT):
                raise RuntimeError("dask scheduler did not start")
            worker = subprocess.Popen(
                ["uv", "run", "--project", str(REPO), "dask", "worker",
                 f"tcp://127.0.0.1:{SCHED_PORT}", "--memory-limit", "8GB",
                 "--nworkers", "1", "--no-dashboard"],
                cwd=DEV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            procs.append(worker)
            env["TIDE_BENCH_SCHED"] = f"tcp://127.0.0.1:{SCHED_PORT}"
            time.sleep(3)
            dask_pids = [sched.pid, worker.pid]

        gunicorn = subprocess.Popen(
            ["uv", "run", "--project", str(REPO), "gunicorn", "w8app:app",
             "-w", "2", "-k", "uvicorn.workers.UvicornWorker",
             "-b", f"127.0.0.1:{PORT}", "--timeout", "180"],
            cwd=DEV, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        procs.append(gunicorn)
        if not wait_port(PORT):
            raise RuntimeError("gunicorn did not start")
        workers = find_workers(gunicorn.pid)

        # warmup: touch both workers
        for _ in range(8):
            get(f"{BASE}/bench/point?lon=121.5&lat=25.0")
        get(f"{BASE}/bench/map?lon0=120&lon1=121&lat0=23&lat1=24&sample=1")

        sampler = TreeSampler({"gunicorn": workers, "dask": dask_pids})
        sampler.start()
        time.sleep(1.0)
        baseline = sampler.snapshot()
        base_tree = sampler.tree_peak

        rng = np.random.default_rng(20260611)
        r = {"workers": workers}

        pts = []
        for _ in range(40):
            lon = float(rng.uniform(106, 149)); lat = float(rng.uniform(1, 44))
            dt, _, _ = get(f"{BASE}/bench/point?lon={lon}&lat={lat}")
            pts.append(dt)
        r["point_ms_median"] = round(1e3 * float(np.median(pts)), 2)
        r["point_ms_p95"] = round(1e3 * float(np.percentile(pts, 95)), 2)

        q = "&".join(f"{k}={v}" for k, v in BENCH.items())
        for name, qq in (("W6_s5", f"{q}&sample=5"),
                         ("W6b_s5_phase3",
                          f"lon0={105 + 0.1}&lon1={150 + 0.1}&lat0=0.1&lat1=45.1&sample=5")):
            dt, nbytes, _ = get(f"{BASE}/bench/map?{qq}")
            r[name] = {"s": round(dt, 2), "MiB": round(nbytes / 2**20, 1)}

        w8 = []
        for _ in range(3):
            dt, nbytes, _ = get(f"{BASE}/bench/map?{q}&sample=1")
            w8.append((dt, nbytes))
        r["W8_single"] = {"s_median": round(float(np.median([x[0] for x in w8])), 2),
                          "payload_MiB": round(w8[0][1] / 2**20, 1)}

        conc_pids, conc_walls = set(), []
        for _ in range(2):
            out = [None, None]

            def hit(slot):
                t0 = time.perf_counter()
                _, _, pid = get(f"{BASE}/bench/map?{q}&sample=1")
                out[slot] = (time.perf_counter() - t0, pid)

            th = [threading.Thread(target=hit, args=(s,)) for s in (0, 1)]
            t0 = time.perf_counter()
            [t.start() for t in th]; [t.join() for t in th]
            conc_walls.append(time.perf_counter() - t0)
            conc_pids |= {o[1] for o in out}
        r["W8_conc2"] = {"wall_s_max": round(max(conc_walls), 2),
                         "distinct_pids": sorted(conc_pids)}

        sampler.stop(); sampler.join()
        peaks = sampler.snapshot()
        g_peak = max(peaks["gunicorn"].values())
        g_delta = max(peaks["gunicorn"][p] - baseline["gunicorn"][p]
                      for p in peaks["gunicorn"])
        d_peak = max(peaks["dask"].values()) if dask_pids else 0
        d_delta = (max(peaks["dask"][p] - baseline["dask"][p]
                       for p in peaks["dask"]) if dask_pids else 0)
        tree_delta = sampler.tree_peak - base_tree
        headroom = sampler.min_avail / psutil.virtual_memory().total
        r["memory"] = {
            "gunicorn_peak_GiB": round(g_peak / GiB, 2),
            "gunicorn_delta_GiB": round(g_delta / GiB, 2),
            "dask_peak_GiB": round(d_peak / GiB, 2),
            "dask_delta_GiB": round(d_delta / GiB, 2),
            "tree_peak_GiB": round(sampler.tree_peak / GiB, 2),
            "tree_delta_GiB": round(tree_delta / GiB, 2),
            "min_host_headroom_frac": round(headroom, 3),
        }

        fails = []
        if g_delta > THRESH["gunicorn_delta"]: fails.append("gunicorn_delta")
        if g_peak > THRESH["gunicorn_peak"]: fails.append("gunicorn_peak")
        if dask_pids and d_delta > THRESH["dask_delta"]: fails.append("dask_delta")
        if dask_pids and d_peak > THRESH["dask_peak"]: fails.append("dask_peak")
        if tree_delta > THRESH["tree_delta"]: fails.append("tree_delta")
        if headroom < THRESH["headroom_frac"]: fails.append("headroom")
        if w8[0][1] > THRESH["payload"]: fails.append("payload")
        if r["W8_single"]["s_median"] > THRESH["wall_single"]: fails.append("wall_single")
        if r["W8_conc2"]["wall_s_max"] > THRESH["wall_conc2"]: fails.append("wall_conc2")
        if len(conc_pids) < 2: fails.append("distinct_worker_pids")
        r["failures"] = fails
        results[mode] = r
        print(f"== {mode}: point {r['point_ms_median']}ms | "
              f"W6 {r['W6_s5']['s']}s | W8 {r['W8_single']['s_median']}s "
              f"{r['W8_single']['payload_MiB']}MiB | conc2 "
              f"{r['W8_conc2']['wall_s_max']}s pids={len(conc_pids)} | "
              f"gw peak {r['memory']['gunicorn_peak_GiB']}GiB | "
              + ("PASS" if not fails else f"FAIL {fails}"))
    finally:
        for p in reversed(procs):
            try:
                parent = psutil.Process(p.pid)
                for c in parent.children(recursive=True):
                    c.terminate()
                p.terminate()
                p.wait(timeout=15)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        time.sleep(1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--out", type=Path,
                    default=DEV / "benchmarks" / "w8_harness.json")
    args = ap.parse_args()
    results = {"_thresholds_GiB_MiB_s": {k: (v / GiB if "delta" in k or "peak" in k
                                             else v) for k, v in THRESH.items()}}
    for mode in args.modes.split(","):
        run_mode(mode, results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    any_fail = any(results[m].get("failures") for m in results if not m.startswith("_"))
    print(("STOP-AT-G1: threshold failure — re-sign-off required"
           if any_fail else "PASS w8_harness (all signed thresholds hold)")
          + f" -> {args.out}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
