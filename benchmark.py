#!/usr/bin/env python3
"""
benchmark.py — Compare Fused3S v0/v1/v2 kernels vs PyTorch TransformerConv
                vs CPU/GPU baselines on Cora, Citeseer

Usage:
    python benchmark.py                                     # All datasets, all methods
    python benchmark.py --datasets cora arxiv               # Subset of datasets
    python benchmark.py --warmup 5 --iters 20               # Custom iteration counts
    python benchmark.py --build                             # Force rebuild C++ binaries
    python benchmark.py --json results/bench.json           # Export to JSON
    python benchmark.py --skip-cpu-baseline                 # Skip slow CPU baseline on big graphs
    python benchmark.py --skip-cuda-binaries                # Only run TransformerConv
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[0]  # fused3s/
BUILD_DIR = REPO_ROOT / "build"
DATA_DIR = REPO_ROOT / "data"

# Binary map: label -> (cmake_target, binary_name)
FUSED_BINARIES = {
    "fused3s_v0": ("test_fused_3s_v0", BUILD_DIR / "test_fused_3s_v0"),
    "fused3s_v1": ("test_fused_3s_v1", BUILD_DIR / "test_fused_3s_v1"),
    "fused3s_v2": ("test_fused_3s_v2", BUILD_DIR / "test_fused_3s_v2"),
}
BASELINE_TARGET = ("test_baselines", BUILD_DIR / "test_baselines")

# Datasets — directory name under data/
DATASETS = {
    "cora":     DATA_DIR / "cora",
    "citeseer": DATA_DIR / "citeseer",
}

FEAT_DIM = 64

# Kernel time regex patterns (v0/v1 print "Kernel Execution Time:", v2 prints "Kernel Time:")
KERNEL_TIME_PATTERNS = [
    re.compile(r"Kernel(?:\s+Execution)?\s+Time:\s*([\d.]+)\s*ms"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csr_graph(dataset_dir: Path):
    """Load CSR graph from text files (mirrors C++ load_csr_graph)."""
    meta = (dataset_dir / "meta.txt").read_text().strip().split()
    num_nodes, num_edges = int(meta[0]), int(meta[1])

    row_ptr = np.loadtxt(dataset_dir / "row_ptr.txt", dtype=np.int64)
    col_file = dataset_dir / "col_idx.txt"
    if not col_file.exists():
        col_file = dataset_dir / "col_ind.txt"
    col_idx = np.loadtxt(col_file, dtype=np.int64)

    return num_nodes, num_edges, row_ptr, col_idx


def ensure_symlinks():
    """Ensure col_ind.txt symlinks exist for C++ binaries."""
    for name, ddir in DATASETS.items():
        if not ddir.exists():
            continue
        col_idx = ddir / "col_idx.txt"
        col_ind = ddir / "col_ind.txt"
        if col_idx.exists() and not col_ind.exists():
            col_ind.symlink_to("col_idx.txt")


def build_binaries(force=False):
    """Build all C++ test binaries if missing (or forced)."""
    targets_needed = []
    for label, (target, binpath) in FUSED_BINARIES.items():
        if force or not binpath.exists():
            targets_needed.append(target)
    bl_target, bl_bin = BASELINE_TARGET
    if force or not bl_bin.exists():
        targets_needed.append(bl_target)

    if not targets_needed:
        return

    print(f"[BUILD] Building: {targets_needed}")
    BUILD_DIR.mkdir(exist_ok=True)

    cmake_res = subprocess.run(["cmake", str(REPO_ROOT)], cwd=BUILD_DIR,
                               capture_output=True, text=True)
    if cmake_res.returncode != 0:
        print(f"[BUILD] cmake failed:\n{cmake_res.stderr}", file=sys.stderr)
        sys.exit(1)

    make_res = subprocess.run(["make", "-j"] + targets_needed, cwd=BUILD_DIR,
                              capture_output=True, text=True)
    if make_res.returncode != 0:
        print(f"[BUILD] make failed:\n{make_res.stderr}", file=sys.stderr)
        sys.exit(1)
    print("[BUILD] Done.")


def parse_kernel_time(stdout: str) -> float | None:
    """Extract kernel time from C++ binary stdout."""
    for pattern in KERNEL_TIME_PATTERNS:
        m = pattern.search(stdout)
        if m:
            return float(m.group(1))
    return None


def summarize_times(times: list[float]) -> dict:
    if not times:
        return {"error": "no valid timing data"}
    return {
        "mean_ms": float(np.mean(times)),
        "std_ms":  float(np.std(times)),
        "min_ms":  float(np.min(times)),
        "max_ms":  float(np.max(times)),
        "runs":    len(times),
    }


def fmt_result(res: dict) -> str:
    if "error" in res:
        return f"ERROR: {res['error'][:60]}"
    return (f"{res['mean_ms']:.4f} ± {res['std_ms']:.4f} ms  "
            f"(min={res['min_ms']:.4f}, max={res['max_ms']:.4f}, n={res['runs']})")


# ---------------------------------------------------------------------------
# Benchmark runners
# ---------------------------------------------------------------------------

def bench_fused_kernel(label: str, dataset_path: Path, warmup: int, iters: int,
                       skip_cpu: bool = False, timeout: int = 300) -> dict:
    """Run a fused3s test binary (v0/v1/v2) and collect kernel times."""
    _, binpath = FUSED_BINARIES[label]
    if not binpath.exists():
        return {"error": f"{binpath.name} not found"}

    times_ms = []
    for i in range(warmup + iters):
        try:
            cmd = [str(binpath), str(dataset_path)]
            if skip_cpu:
                cmd.append("--skip-cpu")
            result = subprocess.run(
                cmd,
                cwd=REPO_ROOT,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"error": f"timeout ({timeout}s)"}
        if result.returncode != 0:
            return {"error": f"{binpath.name}: {result.stderr[:120]}"}

        t = parse_kernel_time(result.stdout)
        if t is not None and i >= warmup:
            times_ms.append(t)

    return summarize_times(times_ms)


def bench_baselines(dataset_path: Path, warmup: int, iters: int,
                    skip_cpu: bool = False, timeout: int = 600) -> dict:
    """Run test_baselines and parse CPU/GPU times."""
    _, bl_bin = BASELINE_TARGET
    if not bl_bin.exists():
        return {"cpu": {"error": "not built"}, "gpu": {"error": "not built"}}

    cpu_times, gpu_times = [], []
    for i in range(warmup + iters):
        try:
            result = subprocess.run(
                [str(bl_bin), str(dataset_path)],
                cwd=REPO_ROOT,
                capture_output=True, text=True, timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {"cpu": {"error": f"timeout ({timeout}s)"},
                    "gpu": {"error": f"timeout ({timeout}s)"}}
        if result.returncode != 0:
            return {"cpu": {"error": result.stderr[:120]},
                    "gpu": {"error": result.stderr[:120]}}

        if i >= warmup:
            cpu_m = re.search(r"CPU Sparse Attention Time:\s*([\d.]+)\s*ms", result.stdout)
            gpu_m = re.search(r"GPU Naive Time:\s*([\d.]+)\s*ms", result.stdout)
            if cpu_m:
                cpu_times.append(float(cpu_m.group(1)))
            if gpu_m:
                gpu_times.append(float(gpu_m.group(1)))

    cpu_res = {"error": "skipped"} if skip_cpu else summarize_times(cpu_times)
    return {"cpu": cpu_res, "gpu": summarize_times(gpu_times)report}


def bench_transformerconv(dataset_path: Path, warmup: int, iters: int) -> dict:
    """Benchmark PyG TransformerConv (dot-product attention)."""
    try:
        import torch
        from torch_geometric.nn import TransformerConv
    except ImportError:
        return {"error": "torch / torch_geometric not installed"}

    num_nodes, num_edges, row_ptr, col_idx = load_csr_graph(dataset_path)

    # Reconstruct edge_index from CSR — vectorized for large graphs
    counts = np.diff(row_ptr).astype(np.int64)
    src = np.repeat(np.arange(num_nodes, dtype=np.int64), counts)
    dst = col_idx[:int(row_ptr[-1])].copy()
    edge_index = torch.from_numpy(np.stack([src, dst], axis=0)).long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edge_index = edge_index.to(device)

    conv = TransformerConv(
        FEAT_DIM, FEAT_DIM,
        heads=1, concat=False,
        bias=False, root_weight=False,
    ).to(device)
    conv.eval()

    torch.manual_seed(42)
    x = torch.randn(num_nodes, FEAT_DIM, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = conv(x, edge_index)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # Timed
    times_ms = []
    with torch.no_grad():
        for _ in range(iters):
            if device.type == "cuda":
                torch.cuda.synchronize()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                start_ev.record()
                _ = conv(x, edge_index)
                end_ev.record()
                torch.cuda.synchronize()
                times_ms.append(start_ev.elapsed_time(end_ev))
            else:
                t0 = time.perf_counter()
                _ = conv(x, edge_index)
                t1 = time.perf_counter()
                times_ms.append((t1 - t0) * 1000)

    res = summarize_times(times_ms)
    res["device"] = str(device)
    return res


# ---------------------------------------------------------------------------
# Result table
# ---------------------------------------------------------------------------

METHOD_LABELS = [
    ("fused3s_v0",   "Fused3S v0 (per-block softmax)"),
    ("fused3s_v1",   "Fused3S v1 (online softmax)"),
    ("fused3s_v2",   "Fused3S v2 (register-resident)"),
    ("gpu_baseline", "GPU Baseline (naive)"),
    ("cpu_baseline", "CPU Baseline"),
    ("transformerconv", "TransformerConv (PyG)"),
]


def print_table(all_results: dict):
    datasets = list(all_results.keys())
    col_w = 38
    header = f"{'Method':<35}" + "".join(f"{ds:>{col_w}}" for ds in datasets)
    sep = "-" * len(header)

    print(f"\n{'='*len(header)}")
    print("  BENCHMARK RESULTS — Sparse Attention (SDDMM + Softmax + SpMM)")
    print(f"  feat_dim={FEAT_DIM}")
    print(f"{'='*len(header)}")
    print(header)
    print(sep)

    for key, label in METHOD_LABELS:
        row = f"{label:<35}"
        for ds in datasets:
            res = all_results[ds].get(key, {"error": "not run"})
            row += f"{fmt_result(res):>{col_w}}"
        print(row)

    print(sep)

    # Speedups vs best fused kernel (v2)
    for ds in datasets:
        v2 = all_results[ds].get("fused3s_v2", {})
        if "error" in v2 or "mean_ms" not in v2:
            continue
        v2_t = v2["mean_ms"]
        print(f"\n  Speedups for {ds} (vs Fused3S v2 @ {v2_t:.4f} ms):")
        for key, label in METHOD_LABELS:
            if key == "fused3s_v2":
                continue
            other = all_results[ds].get(key, {})
            if "error" not in other and "mean_ms" in other:
                ratio = other["mean_ms"] / v2_t
                direction = "faster" if ratio < 1.0 else "slower"
                if ratio >= 1.0:
                    print(f"    vs {label}: {ratio:.2f}x {direction}")
                else:
                    print(f"    vs {label}: {1.0/ratio:.2f}x {direction}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fused3S Benchmark Suite (v0/v1/v2)")
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Datasets to benchmark (default: all available)")
    parser.add_argument("--warmup", type=int, default=3,
                        help="Warmup iterations (default: 3)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timed iterations (default: 10)")
    parser.add_argument("--build", action="store_true",
                        help="Force rebuild C++ binaries")
    parser.add_argument("--json", type=str, default=None,
                        help="Export results to JSON file")
    parser.add_argument("--skip-cuda-binaries", action="store_true",
                        help="Skip all C++ CUDA binaries (only run TransformerConv)")
    parser.add_argument("--skip-cpu-baseline", action="store_true",
                        help="Skip CPU baseline (useful for large graphs)")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-run timeout in seconds (default: 300)")
    args = parser.parse_args()

    os.chdir(REPO_ROOT)
    ensure_symlinks()

    # Determine which datasets to run
    if args.datasets:
        ds_to_run = args.datasets
    else:
        ds_to_run = [name for name, path in DATASETS.items() if path.exists()]

    if not ds_to_run:
        print("[ERROR] No datasets found. Run data/data_loader.py first.")
        sys.exit(1)

    # Build C++ binaries
    if not args.skip_cuda_binaries:
        build_binaries(force=args.build)

    all_results = {}

    for ds_name in ds_to_run:
        ds_path = DATASETS.get(ds_name)
        if ds_path is None or not ds_path.exists():
            print(f"[SKIP] Dataset '{ds_name}' not found at {DATASETS.get(ds_name, '?')}")
            continue

        num_nodes, num_edges, _, _ = load_csr_graph(ds_path)
        is_large = num_nodes > 50000

        print(f"\n{'='*70}")
        print(f"  Dataset: {ds_name}  (nodes={num_nodes:,}, edges={num_edges:,})")
        print(f"  warmup={args.warmup}, iters={args.iters}, timeout={args.timeout}s")
        if is_large:
            print(f"  ⚡ Large graph — CPU baseline {'skipped' if args.skip_cpu_baseline else 'may be slow'}")
        print(f"{'='*70}")

        results = {}
        step = 1
        total_steps = 6 if not args.skip_cuda_binaries else 1
        # --- Fused3S v0 / v1 / v2 ---
        if not args.skip_cuda_binaries:
            for version in ["fused3s_v0", "fused3s_v1", "fused3s_v2"]:
                vlabel = version.replace("fused3s_", "").upper()
                print(f"\n[{step}/{total_steps}] Fused3S {vlabel}...")
                results[version] = bench_fused_kernel(
                    version, ds_path, args.warmup, args.iters,
                    skip_cpu=args.skip_cpu_baseline, timeout=args.timeout)
                print(f"      {fmt_result(results[version])}")
                step += 1

            # --- CPU / GPU baselines ---
            print(f"\n[{step}/{total_steps}] CPU Baseline...")
            step += 1
            print(f"[{step}/{total_steps}] GPU Baseline (naive)...")
            step += 1
            bl = bench_baselines(ds_path, args.warmup, args.iters,
                                 skip_cpu=args.skip_cpu_baseline, timeout=args.timeout)
            results["cpu_baseline"] = bl["cpu"]
            results["gpu_baseline"] = bl["gpu"]
            print(f"  CPU: {fmt_result(results['cpu_baseline'])}")
            print(f"  GPU: {fmt_result(results['gpu_baseline'])}")

        # --- TransformerConv ---
        print(f"\n[{step}/{total_steps}] TransformerConv (PyG)...")
        results["transformerconv"] = bench_transformerconv(ds_path, args.warmup, args.iters)
        print(f"      {fmt_result(results['transformerconv'])}")

        all_results[ds_name] = results

    # Summary table
    if all_results:
        print_table(all_results)

    # JSON export
    if args.json:
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            return obj

        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, default=convert)
        print(f"[JSON] Results saved to {out_path}")


if __name__ == "__main__":
    main()
