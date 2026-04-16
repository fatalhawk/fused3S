<p align="center">
  <h1 align="center">⚡ Fused3S</h1>
  <p align="center">
    <strong>Fused Sparse-Dense Graph Attention on Tensor Cores</strong>
  </p>
  <p align="center">
    A high-performance CUDA kernel that fuses <b>SDDMM + Softmax + SpMM</b> into a single pass<br/>
    using NVIDIA Tensor Cores (WMMA) for sparse graph transformer attention.
  </p>
  <p align="center">
    <a href="#key-ideas">Key Ideas</a> •
    <a href="#kernel-versions">Kernel Versions</a> •
    <a href="#project-structure">Project Structure</a> •
    <a href="#getting-started">Getting Started</a> •
    <a href="#benchmarks">Benchmarks</a>
  </p>
</p>

---

## Overview

Graph transformers rely on sparse attention to capture structural relationships, but the standard three-kernel pipeline — **Sampled Dense-Dense Matrix Multiply (SDDMM)**, **Softmax**, and **Sparse Matrix-Matrix Multiply (SpMM)** — incurs significant memory traffic between stages. **Fused3S** eliminates these intermediate materializations by fusing all three stages into a single GPU kernel that operates on a **block-sparse** representation of the graph adjacency.

The kernel processes the graph in **16 × 16 Row Windows** aligned with the native WMMA tile size, using the block-sparse format derived from the CSR adjacency. Each thread block handles one Row Window, and load balancing is achieved by sorting windows by workload weight.

## Key Ideas

| Concept | Description |
|---|---|
| **Block-Sparse Format** | CSR adjacency is converted to 16×16 block tiles. Each Row Window maps to a set of active column blocks. |
| **Fused 3-Stage Pipeline** | SDDMM → Softmax → SpMM occurs inside a single kernel, with data flowing through shared memory / registers instead of global memory. |
| **Tensor Core Acceleration** | All matrix multiplications (Q×Kᵀ and P×V) use `wmma::mma_sync` on 16×16×16 FP16 tiles. |
| **Online Softmax** | A streaming softmax algorithm maintains running max and sum across column blocks, enabling numerically correct normalization without a separate pass. |
| **Q Caching** | The Query tile for each Row Window is loaded once into shared memory and reused across all column blocks. |
| **Load Balancing** | Row Windows are sorted in descending order by active block count, giving heavier windows to earlier SMs. |

## Kernel Versions

The project tracks three progressive optimization stages:

### `v0` — Per-Block Softmax + 1D Thread Layout
> Baseline fused kernel. Each warp independently processes column blocks with a local softmax. Simple and functional but softmax is only correct _within_ individual 16×16 blocks — not across the full row.

### `v1` — Online Softmax + 2D Thread Blocks
> Introduces the online (streaming) softmax algorithm that maintains running max / sum state across all column blocks, producing globally correct attention weights. Uses a 2D thread layout `dim3(32, WARPS_PER_BLOCK)` where `threadIdx.y` directly maps to the warp ID.

### `v2` — Register-Resident Output + Fragment Rescaling
> Moves the output accumulator (`frag_O`) into WMMA register fragments for the entire kernel lifetime, eliminating the per-iteration shared memory round-trip for `smem_O[16][64]`. Rescaling uses the known sm_80+ WMMA fragment layout to directly multiply register elements by per-row correction factors. Also removes one `__syncthreads()` per iteration since SpMM only writes to registers.

```
v0 (baseline fused) → v1 (correct softmax) → v2 (register-resident O)
```

## Project Structure

```
fused3S/
├── src/
│   ├── fused3s_v0.cu          # v0: Per-block softmax kernel
│   ├── fused3s_v1.cu          # v1: Online softmax + 2D layout
│   ├── fused3s_v2.cu          # v2: Register-resident O + fragment rescaling
│   └── converter.cu           # CSR → Block-Sparse format converter
├── baselines/
│   ├── cpu_baseline.cpp       # Naive CPU sparse attention (ground truth)
│   └── gpu_baseline.cu        # Naive GPU sparse attention (1 thread/node)
├── tests/
│   ├── test_baselines.cu      # Baseline correctness tests
│   ├── test_fused_3s_v0.cu    # v0 test + CPU verification
│   ├── test_fused_3s_v1.cu    # v1 test + CPU verification
│   └── test_fused_3s_v2.cu    # v2 test + CPU verification
├── include/
│   └── utils.cuh              # Graph loading, block-sparse format, load balancing
├── data/
│   ├── data_loader.py         # PyG → CSR export script (Cora, Citeseer)
│   ├── cora/                  # Cora citation dataset (CSR files)
│   └── citeseer/              # Citeseer citation dataset (CSR files)
├── profiles/                  # NVIDIA Nsight Compute profiling reports (.ncu-rep)
├── CMakeLists.txt             # Build configuration (CUDA sm_89, C++17)
└── requirements.txt           # Python dependencies for data loading
```

## Getting Started

### Prerequisites

- **CUDA Toolkit** ≥ 11.0 (Tensor Core WMMA support)
- **GPU**: Compute Capability ≥ `sm_80` (Ampere or newer; configured for `sm_89`)
- **CMake** ≥ 3.24
- **C++17** compatible compiler
- **Python 3.8+** with PyTorch Geometric (for data loading only)

### 1. Prepare Graph Data

```bash
pip install -r requirements.txt
cd data
python data_loader.py
```

This downloads the **Cora** and **Citeseer** citation graphs via PyTorch Geometric and exports them as CSR text files (`meta.txt`, `row_ptr.txt`, `col_idx.txt`).

### 2. Build

```bash
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

This produces three test executables: `test_fused_3s_v0`, `test_fused_3s_v1`, `test_fused_3s_v2`, and `test_baselines`.

### 3. Run & Verify

```bash
# Run with default dataset (Cora)
./test_fused_3s_v2

# Run with a specific dataset
./test_fused_3s_v2 data/citeseer
```

Each test binary:
1. Loads the CSR graph and converts it to block-sparse format
2. Generates random Q, K, V matrices (FP32 → FP16)
3. Computes CPU ground truth using the **exact same block-sparse semantics**
4. Launches the fused kernel and compares GPU output against CPU
5. Reports `SUCCESS` / `FAILURE` and kernel execution time

### 4. Profile

```bash
ncu --set full -o profiles/v2_cora ./test_fused_3s_v2 data/cora
```

Pre-captured `.ncu-rep` reports are available in the `profiles/` directory for all kernel versions on both datasets.

## Benchmarks

Results on **Cora** (2,708 nodes, 10,556 edges, 170 Row Windows) using RTX 4090 (sm_89):

| Kernel | Execution Time | Softmax Scope | Output Storage |
|---|---|---|---|
| `v0` | ~0.26 ms | Per-block (local) | Global memory directly |
| `v1` | ~0.43 ms | Global (online) | Shared memory |
| `v2` | ~0.26 ms | Global (online) | Register fragments |

## Technical Details

### Block-Sparse Data Flow

```
CSR Adjacency (row_ptr, col_ind)
        │
        ▼
┌─────────────────────┐
│  CSR → Block-Sparse │   converter.cu
│  (16×16 tiles)      │
└─────────────────────┘
        │
        ▼
  window_offsets[]        ← start/end of each Row Window's column blocks
  block_col_indices[]     ← which 16×16 column blocks are active
  window_order[]          ← sorted by workload for load balancing
```

### Fused Kernel Pipeline (per Row Window)

```
 ┌──────────────┐     ┌────────────┐     ┌──────────────┐
 │  SDDMM       │     │  Softmax   │     │  SpMM        │
 │  Q × Kᵀ      │ ──▶ │  (online)  │ ──▶ │  P × V       │
 │  via WMMA     │     │  smem/regs │     │  via WMMA    │
 └──────────────┘     └────────────┘     └──────────────┘
        ▲                                       │
        │          shared memory / registers     │
        └───────────── no global memory ─────────┘
```

## Author

**Mayank Choudhary**

CS299 — Independent Study

## License

This project is for academic research purposes.