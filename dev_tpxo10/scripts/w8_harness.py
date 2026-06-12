#!/usr/bin/env python
"""§7.5.3 W8 harness driver v2 (rounds 12 fixes: true process-tree RSS,
header-identified gunicorn workers, cap sweep).

Run in the dev env; servers run in the PRODUCTION env via
`uv run --project <repo>`. w8app reproduces the production /api/tide map
serialization path verbatim (see w8app.py).

Modes:
  --modes direct,dask,dist-auto,dist-native    full W8 threshold runs
  --schema tpxo10|legacy                       store schema under test
  --sweep 250000,400000,...                    cap sweep (direct mode):
        square bboxes sized to N output cells at sample=1, each run as
        single request x2 + concurrency-2 x1 with full memory sampling

Memory sampling (round 12, finding 1): the sampler re-walks each process
GROUP'S full recursive tree every 50 ms (uv-run launchers spawn the real
gunicorn master / dask scheduler / nanny / worker as children), tracks
per-PID first-seen baseline and peak, and aggregates the whole tree.
Gunicorn WORKER processes are identified by X-Worker-PID response
headers, never by child enumeration alone.
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
    """50 ms sampler. Groups are ROOT pids; every tick re-walks each
    root's recursive children, so late-spawned processes (gunicorn
    master/workers under uv-run, dask nanny/worker) are captured. Tracks
    per-PID first-seen baseline + peak, and the aggregate tree peak."""

    def __init__(self, roots: dict[str, list[int]]):
        super().__init__(daemon=True)
        self.roots = roots
        self.base: dict[str, dict[int, int]] = {k: {} for k in roots}
        self.peak: dict[str, dict[int, int]] = {k: {} for k in roots}
        self.tree_base = None
        self.tree_peak = 0
        self.min_avail = psutil.virtual_memory().available
        self._halt = threading.Event()

    def _walk(self):
        out = {}
        for name, rootpids in self.roots.items():
            pids = {}
            for rp in rootpids:
                try:
                    p = psutil.Process(rp)
                    procs = [p] + p.children(recursive=True)
                except psutil.Error:
                    continue
                for c in procs:
                    try:
                        pids[c.pid] = c.memory_info().rss
                    except psutil.Error:
                        continue
            out[name] = pids
        return out

    def run(self):
        while not self._halt.is_set():
            snap = self._walk()
            total = 0
            for name, pids in snap.items():
                for pid, rss in pids.items():
                    self.base[name].setdefault(pid, rss)
                    self.peak[name][pid] = max(self.peak[name].get(pid, 0), rss)
                    total += rss
            if self.tree_base is None:
                self.tree_base = total
            self.tree_peak = max(self.tree_peak, total)
            self.min_avail = min(self.min_avail,
                                 psutil.virtual_memory().available)
            time.sleep(0.05)

    def stats(self, name: str, pid_filter=None):
        pids = self.peak.get(name, {})
        if pid_filter is not None:
            pids = {p: v for p, v in pids.items() if p in pid_filter}
        if not pids:
            return 0, 0
        peak = max(pids.values())
        delta = max(v - self.base[name].get(p, v) for p, v in pids.items())
        return peak, delta

    def stop(self):
        self._halt.set()


def wait_port(port, timeout=120):
    t0 = time.time()
    while time.time() - t0 < timeout:
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return True
        time.sleep(0.5)
    return False


def get(url):
    t0 = time.perf_counter()
    with urllib.request.urlopen(url, timeout=300) as r:
        body = r.read()
        pid = r.headers.get("X-Worker-PID")
    return time.perf_counter() - t0, len(body), int(pid) if pid else None


def eval_thresholds(r, has_dask):
    fails = []
    m = r["memory"]
    if m["gunicorn_delta_GiB"] * GiB > THRESH["gunicorn_delta"]: fails.append("gunicorn_delta")
    if m["gunicorn_peak_GiB"] * GiB > THRESH["gunicorn_peak"]: fails.append("gunicorn_peak")
    if has_dask and m["dask_delta_GiB"] * GiB > THRESH["dask_delta"]: fails.append("dask_delta")
    if has_dask and m["dask_peak_GiB"] * GiB > THRESH["dask_peak"]: fails.append("dask_peak")
    if m["tree_delta_GiB"] * GiB > THRESH["tree_delta"]: fails.append("tree_delta")
    if m["min_host_headroom_frac"] < THRESH["headroom_frac"]: fails.append("headroom")
    if r["W8_single"]["payload_MiB"] * 2**20 > THRESH["payload"]: fails.append("payload")
    if r["W8_single"]["s_median"] > THRESH["wall_single"]: fails.append("wall_single")
    if r["W8_conc2"]["wall_s_max"] > THRESH["wall_conc2"]: fails.append("wall_conc2")
    if r["W8_conc2"]["n_distinct_pids"] < 2: fails.append("distinct_worker_pids")
    return fails


class Service:
    """Production-like service stack for one mode."""

    def __init__(self, mode: str, schema: str):
        self.mode, self.schema = mode, schema
        self.procs: list[subprocess.Popen] = []
        self.dask_roots: list[int] = []
        self.worker_pids: set[int] = set()

    def __enter__(self):
        env = os.environ.copy()
        env["TIDE_BENCH_OPEN"] = self.mode
        env["TIDE_BENCH_SCHEMA"] = self.schema
        if self.mode.startswith("dist"):
            sched = subprocess.Popen(
                ["uv", "run", "--project", str(REPO), "dask", "scheduler",
                 "--port", str(SCHED_PORT), "--no-dashboard"],
                cwd=DEV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.procs.append(sched)
            if not wait_port(SCHED_PORT):
                raise RuntimeError("dask scheduler did not start")
            worker = subprocess.Popen(
                ["uv", "run", "--project", str(REPO), "dask", "worker",
                 f"tcp://127.0.0.1:{SCHED_PORT}", "--memory-limit", "8GB",
                 "--nworkers", "1", "--no-dashboard"],
                cwd=DEV, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.procs.append(worker)
            env["TIDE_BENCH_SCHED"] = f"tcp://127.0.0.1:{SCHED_PORT}"
            time.sleep(3)
            self.dask_roots = [sched.pid, worker.pid]

        gunicorn = subprocess.Popen(
            ["uv", "run", "--project", str(REPO), "gunicorn", "w8app:app",
             "-w", "2", "-k", "uvicorn.workers.UvicornWorker",
             "-b", f"127.0.0.1:{PORT}", "--timeout", "300"],
            cwd=DEV, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs.append(gunicorn)
        self.gunicorn_root = gunicorn.pid
        if not wait_port(PORT):
            raise RuntimeError("gunicorn did not start")
        # identify WORKERS by response header PIDs (round 12, finding 1)
        for _ in range(24):
            _, _, pid = get(f"{BASE}/bench/point?lon=121.5&lat=25.0")
            self.worker_pids.add(pid)
            if len(self.worker_pids) >= 2:
                break
        get(f"{BASE}/bench/map?lon0=120&lon1=121&lat0=23&lat1=24&sample=1")
        return self

    def __exit__(self, *exc):
        for p in reversed(self.procs):
            try:
                parent = psutil.Process(p.pid)
                for c in parent.children(recursive=True):
                    c.terminate()
                p.terminate()
                p.wait(timeout=20)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        time.sleep(1)


def measure_w8(svc: Service, bench: dict, n_single=3, n_conc=2):
    sampler = TreeSampler({"gunicorn": [svc.gunicorn_root],
                           "dask": svc.dask_roots})
    sampler.start()
    time.sleep(0.5)
    r = {}
    q = "&".join(f"{k}={v}" for k, v in bench.items())

    w8 = [get(f"{BASE}/bench/map?{q}&sample=1") for _ in range(n_single)]
    r["W8_single"] = {
        "s_median": round(float(np.median([x[0] for x in w8])), 2),
        "payload_MiB": round(w8[0][1] / 2**20, 1)}

    conc_pids, conc_walls = set(), []
    for _ in range(n_conc):
        out = [None, None]

        def hit(slot):
            _, _, pid = get(f"{BASE}/bench/map?{q}&sample=1")
            out[slot] = pid

        th = [threading.Thread(target=hit, args=(s,)) for s in (0, 1)]
        t0 = time.perf_counter()
        [t.start() for t in th]; [t.join() for t in th]
        conc_walls.append(time.perf_counter() - t0)
        conc_pids |= set(out)
    r["W8_conc2"] = {"wall_s_max": round(max(conc_walls), 2),
                     "n_distinct_pids": len(conc_pids)}

    sampler.stop(); sampler.join()
    g_peak, g_delta = sampler.stats("gunicorn", pid_filter=svc.worker_pids)
    d_peak, d_delta = sampler.stats("dask")
    r["memory"] = {
        "gunicorn_peak_GiB": round(g_peak / GiB, 2),
        "gunicorn_delta_GiB": round(g_delta / GiB, 2),
        "dask_peak_GiB": round(d_peak / GiB, 2),
        "dask_delta_GiB": round(d_delta / GiB, 2),
        "tree_peak_GiB": round(sampler.tree_peak / GiB, 2),
        "tree_delta_GiB": round((sampler.tree_peak - (sampler.tree_base or 0)) / GiB, 2),
        "min_host_headroom_frac": round(
            sampler.min_avail / psutil.virtual_memory().total, 3),
        "n_dask_pids_sampled": len(sampler.peak.get("dask", {})),
    }
    return r


def run_mode(mode: str, schema: str, results: dict):
    with Service(mode, schema) as svc:
        rng = np.random.default_rng(20260611)
        pts = []
        for _ in range(40):
            lon = float(rng.uniform(106, 149)); lat = float(rng.uniform(1, 44))
            dt, _, _ = get(f"{BASE}/bench/point?lon={lon}&lat={lat}")
            pts.append(dt)
        extra = {"point_ms_median": round(1e3 * float(np.median(pts)), 2),
                 "point_ms_p95": round(1e3 * float(np.percentile(pts, 95)), 2)}
        q = "&".join(f"{k}={v}" for k, v in BENCH.items())
        for name, qq in (("W6_s5", f"{q}&sample=5"),
                         ("W6b_s5_phase3",
                          "lon0=105.1&lon1=150.1&lat0=0.1&lat1=45.1&sample=5")):
            dt, nbytes, _ = get(f"{BASE}/bench/map?{qq}")
            extra[name] = {"s": round(dt, 2), "MiB": round(nbytes / 2**20, 1)}

        r = measure_w8(svc, BENCH)
        r.update(extra)
        r["worker_pids"] = sorted(svc.worker_pids)
        r["failures"] = eval_thresholds(r, bool(svc.dask_roots))
        results[f"{schema}/{mode}"] = r
        print(f"== {schema}/{mode}: point {r['point_ms_median']}ms | "
              f"W6 {r['W6_s5']['s']}s/{r['W6b_s5_phase3']['s']}s | "
              f"W8 {r['W8_single']['s_median']}s {r['W8_single']['payload_MiB']}MiB | "
              f"conc2 {r['W8_conc2']['wall_s_max']}s pids={r['W8_conc2']['n_distinct_pids']} | "
              f"gw {r['memory']['gunicorn_peak_GiB']}/{r['memory']['gunicorn_delta_GiB']}GiB "
              f"dask {r['memory']['dask_peak_GiB']}GiB({r['memory']['n_dask_pids_sampled']}p) "
              f"tree {r['memory']['tree_delta_GiB']}GiB | "
              + ("PASS" if not r["failures"] else f"FAIL {r['failures']}"))


def run_sweep(cells_list: list[int], schema: str, results: dict):
    """Cap sweep (round 12, finding 3): fresh service per candidate so
    allocator high-water marks don't leak across candidates."""
    for cells in cells_list:
        side_deg = float(np.sqrt(cells)) / 30.0
        bench = dict(lon0=105.0, lon1=round(105.0 + side_deg, 4),
                     lat0=0.0, lat1=round(0.0 + side_deg, 4))
        with Service("direct", schema) as svc:
            r = measure_w8(svc, bench, n_single=2, n_conc=1)
            r["cells_target"] = cells
            r["bbox"] = bench
            r["failures"] = eval_thresholds(r, False)
            results[f"sweep/{schema}/{cells}"] = r
            print(f"== sweep {cells:>7d} cells ({side_deg:.1f}°): "
                  f"W8 {r['W8_single']['s_median']}s "
                  f"{r['W8_single']['payload_MiB']}MiB | conc2 "
                  f"{r['W8_conc2']['wall_s_max']}s | gw "
                  f"{r['memory']['gunicorn_peak_GiB']}/{r['memory']['gunicorn_delta_GiB']}GiB "
                  f"tree {r['memory']['tree_delta_GiB']}GiB | "
                  + ("PASS" if not r["failures"] else f"FAIL {r['failures']}"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--modes", type=str, default="")
    ap.add_argument("--schema", type=str, default="tpxo10")
    ap.add_argument("--sweep", type=str, default="")
    ap.add_argument("--out", type=Path,
                    default=DEV / "benchmarks" / "w8_harness_v2.json")
    args = ap.parse_args()
    results = {"_thresholds": {k: round(v / GiB, 2) if "delta" in k or "peak" in k
                               else v for k, v in THRESH.items()},
               "_serialization": "production-verbatim tide_to_output + "
                                 "jsonable_encoder + ORJSONResponse"}
    for mode in [m for m in args.modes.split(",") if m]:
        run_mode(mode, args.schema, results)
    if args.sweep:
        run_sweep([int(x) for x in args.sweep.split(",")], args.schema, results)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        old = json.loads(args.out.read_text())
        old.update(results)
        results = old
    args.out.write_text(json.dumps(results, indent=1))
    any_fail = any(v.get("failures") for k, v in results.items()
                   if isinstance(v, dict) and not k.startswith("_")
                   and not k.startswith("sweep/"))
    print(("STOP-AT-G1 (full-W8 threshold failure)" if any_fail
           else "w8_harness_v2 complete") + f" -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
