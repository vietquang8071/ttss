# Gaussian–Laplacian Zero-Crossing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gaussian → Laplacian → zero-crossing edge detector to the existing C/OpenMP benchmark and preview pipeline.

**Architecture:** C computes the four stages in both sequential and OpenMP variants. The program benchmarks the complete pipeline, compares signed Laplacian responses with the current tolerance, and checks edge maps exactly. Existing scripts save both outputs and previews show the edge map beside the source image.

**Tech Stack:** C11, OpenMP, Python 3, NumPy, Pillow, `unittest`.

**Spec:** `docs/superpowers/specs/2026-11-20-log-zero-crossing-design.md`; detailed execution requirements are in the user's 2026-11-20 implementation plan.

## Global Constraints

- Use a separable 5-tap Gaussian kernel (radius 2), default sigma 1.0, configurable via CLI, with clamped-edge padding.
- Keep the existing 4-neighbor Laplacian kernel, zero padding, and signed float32 output.
- Check all 8 neighbors for opposite nonzero signs satisfying `abs(p) + abs(q) >= threshold`; default threshold is 10.0 and is configurable via CLI.
- Mark both pixels of qualifying pairs as 1.0; otherwise 0.0. Only evaluate pairs where both pixels are interior.
- Implement Gaussian horizontal, Gaussian vertical, Laplacian, and zero-crossing stages in sequential and OpenMP variants; each OpenMP thread writes only its assigned output pixels.
- Benchmark the complete pipeline, excluding file I/O and Gaussian coefficient initialization. Preserve CSV layout, thread counts, speedup, and efficiency calculations.
- Keep `--output` for signed Laplacian float32 and add `--edges-output` for the float32 binary map.
- Do not alter grayscale preprocessing or add runtime dependencies. Preserve all existing uncommitted changes. Do not commit or push.

## Review Focus

- Size 1 has no eligible interior zero-crossing pairs; edge output must be all zero.
- Zero or non-finite sigma and negative/non-finite thresholds must fail cleanly before benchmarking.
- A crossing at the image boundary must not mark a border pixel; both endpoints must be interior.
- Weak sign changes below threshold must be suppressed; threshold equality must be accepted.
- OpenMP runs at each configured thread count must match sequential signed responses within tolerance and edge maps exactly, with each thread writing disjoint output pixels.

---

### Task 1: Specify numerical expectations in tests

**Files:**
- Modify: `laplacian_openmp/test_laplacian.py`

**Interfaces:**
- Test CLI: `laplacian_benchmark INPUT.bin SIZE [--output SIGNED.bin] [--edges-output EDGES.bin] [--sigma SIGMA] [--threshold THRESHOLD]`.
- Reference: independent Python float32 implementation of the 5-tap separable Gaussian, zero-padded Laplacian, and 8-neighbor thresholded crossing rule.

- [x] Add tests for default/custom sigma, smoothed Laplacian values, sign changes, threshold below/at/above the response-sum boundary, and border exclusion.
- [x] Verify size-1 and constant-image edge maps are empty, and verify a synthetic step produces expected interior edge pixels.
- [x] Run `python3 -m unittest laplacian_openmp.test_laplacian -v`; saw the expected RED before implementation and PASS after implementation.

### Task 2: Implement and benchmark the C/OpenMP pipeline

**Files:**
- Modify: `laplacian_openmp/laplacian_benchmark.c`
- Modify: `laplacian_openmp/test_laplacian.py`

**Interfaces:**
- CLI defaults: sigma `1.0`, threshold `10.0`; accepted options can appear after `INPUT SIZE`.
- Optional outputs: `--output` writes signed float32 Laplacian response; `--edges-output` writes float32 values 0.0/1.0.
- Benchmark rows remain one sequential row plus OpenMP thread counts 2/4/8/12/16, retaining the current CSV columns.

- [x] Implement Gaussian coefficient initialization once, outside timing, and validate sigma/threshold inputs.
- [x] Implement clamped-edge horizontal/vertical Gaussian passes, existing zero-padded Laplacian, and race-free interior zero-crossing output for both sequential and OpenMP paths.
- [x] Time all four stages together; compare sequential/OpenMP Laplacian outputs with the existing tolerance and edge maps exactly; report `PASS` only when both comparisons pass.
- [x] Reject `--output` and `--edges-output` destinations that are the same path or existing filesystem aliases before benchmarking.
- [x] Save requested outputs outside timed regions and release all allocated buffers on success and error paths.
- [x] Rerun `python3 -m unittest laplacian_openmp.test_laplacian -v` and confirm it passes.

### Task 3: Integrate edge outputs into the single-image and dataset preview paths

**Files:**
- Modify: `laplacian_openmp/visualize.py`
- Modify: `laplacian_openmp/test_one.sh`
- Modify: `laplacian_openmp/run_all.py`
- Modify: `laplacian_openmp/run_full.sh`
- Modify: `laplacian_openmp/test_run_all_export.py`

**Interfaces:**
- `visualize.py` consumes the grayscale input and float32 edge map; preview remains a side-by-side PNG.
- `test_one.sh` and `run_all.py` request both C outputs when generating previews; dataset CSV row validation matches the existing six-row benchmark output.

- [x] Add integration tests that verify preview pixels represent the binary edge map and that generated result directories contain no binary intermediates.
- [x] Update single-image and dataset scripts to save/pass both outputs, while keeping preview creation outside benchmark timing.
- [x] Fix and test the CSV row-count expectation without changing the CSV schema or OpenMP thread set.
- [x] Run `python3 -m unittest laplacian_openmp.test_run_all_export -v` and the C test module.

### Task 4: Document the algorithm and verify the full pipeline

**Files:**
- Modify: `README.md`
- Modify: `laplacian_openmp/README.md`

- [x] Document default sigma/threshold, CLI overrides, padding, output file semantics, edge-map previews, and benchmark scope in both READMEs.
- [x] Run the complete project test suite: `python3 -m unittest discover -s laplacian_openmp -p 'test_*.py' -v`.
- [x] Compile with `make -C laplacian_openmp clean all` and run small end-to-end benchmarks through single-image and full-dataset scripts using temporary synthetic grayscale `.bin` inputs.
- [x] Verify CSV rows, output dimensions/values, preview image dimensions/content, and correctness status; inspect `git diff` and `git status --short` to confirm pre-existing edits remain intact.
