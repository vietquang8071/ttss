# Gaussian Laplacian Zero-Crossing Design

## Intent

Extend the existing C/OpenMP image pipeline so it smooths each grayscale image with a separable Gaussian, computes its signed Laplacian response, and detects edges at zero crossings. Benchmark the full per-pixel pipeline and render the resulting edge map in the existing previews.

## Goals and constraints

- Use a 5-tap separable Gaussian kernel with radius 2 and default sigma 1.0; expose sigma as a positive CLI option.
- Keep the current four-neighbor Laplacian stencil. Use the signed response of the smoothed image as a diagnostic output.
- Detect a crossing when an interior pixel and one of its eight neighbors have opposite nonzero signs and the sum of their absolute responses meets a configurable contrast threshold. Default threshold: 10.0, in the input's grayscale 0–255 response units. Mark both pixels of a qualifying pair in the float32 edge map (1.0 for edge, 0.0 otherwise).
- Apply clamped-edge padding in the Gaussian passes. Preserve zero padding for the Laplacian; only pixels with a complete 8-neighbor neighborhood are eligible for zero-crossing output.
- Run Gaussian horizontal/vertical passes, Laplacian, and crossing detection in both sequential and OpenMP variants. Benchmark all four stages per image, excluding file I/O and one-time Gaussian coefficient setup. Correctness requires both Laplacian agreement within the existing tolerance and exact edge-map agreement.
- Keep `--output` as the signed float32 response output and add `--edges-output` for the float32 edge map. The preview displays the input beside the edge map. Integrate both single-image and full-dataset scripts.
- Add no runtime dependencies. Update the root and `laplacian_openmp` READMEs; preserve existing uncommitted thread-count changes.

## Non-goals

- No change to image loading or grayscale preprocessing.
- No alternate Laplacian kernels, automatic threshold selection, or multiscale detection.
- No change to the existing sequential/OpenMP thread-count set or CSV row layout.

## Approaches considered

1. **Full C/OpenMP pipeline (selected):** benchmark smoothing, Laplacian, and zero-crossing together; produce a diagnostic response and an edge map. This meets the request and keeps performance measurements representative of the new algorithm.
2. **Python preview-only processing:** add smoothing and crossing detection only during PNG export. This would leave the C benchmark measuring the old Laplacian-only operation, so it does not meet the requested pipeline integration.

## Builder choices

Use the explicit response-sum threshold rule above so results are deterministic and the threshold can be tuned from the command line. Keep the current zero-padded Laplacian semantics for compatibility, while excluding the outermost border from edge-map detection to avoid crossings without a complete neighborhood.
