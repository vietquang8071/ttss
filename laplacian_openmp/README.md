# Gaussian–Laplacian Zero-Crossing — C/OpenMP

The benchmark runs a four-stage edge-detection pipeline:

1. Separable 5-tap Gaussian smoothing (radius 2, default σ=1.0, clamped-edge padding).
2. The existing four-neighbor Laplacian on the smoothed image, with zero padding:

   ```text
    0  1  0
    1 -4  1
    0  1  0
   ```

3. Zero-crossing detection over all 8 neighbors. A pair qualifies when both pixels are interior, their nonzero Laplacian responses have opposite signs, and `abs(p) + abs(q) >= threshold` (default threshold 10.0).
4. A float32 binary edge map: `1.0` for both pixels in a qualifying pair, `0.0` otherwise. Border pixels remain zero.

The Gaussian sigma and contrast threshold are configurable. The input grayscale range is 0–255, so threshold values use the signed Laplacian response units derived from that range.

## Input format and build

Input `.bin` files have no header and contain **little-endian IEEE-754 float32** values in row-major order for a square `SIZE x SIZE` image. `scripts/preprocess.py` creates these files under `data/processed/{512,1024,2048}/`.

Build on Linux with GCC and OpenMP:

```bash
make -C laplacian_openmp
# or from this directory:
make
```

The program checks the input file size and reports a clear error if `SIZE` does not match. It reads preprocessed `.bin` files, not JPG/PNG files.

## Run one image

From the repository root:

```bash
./laplacian_openmp/laplacian_benchmark data/processed/1024/image_01.bin 1024
./laplacian_openmp/laplacian_benchmark data/processed/1024/image_01.bin 1024 \
  --sigma 1.2 --threshold 14 \
  --output response.bin --edges-output edges.bin
```

Options may follow `INPUT.bin SIZE` in any order:

- `--sigma SIGMA`: positive finite value; default `1.0`.
- `--threshold VALUE`: nonnegative finite value; default `10.0`.
- `--output SIGNED.bin`: optional signed float32 Laplacian response of the Gaussian-smoothed image.
- `--edges-output EDGES.bin`: optional float32 edge map containing only `0.0` and `1.0`.

The program emits one Sequential CSV row followed by five OpenMP rows for 2/4/8/12/16 threads. It warms up each configuration once, times the complete Gaussian + Laplacian + zero-crossing pipeline 10 times, and reports median milliseconds. File I/O and one-time Gaussian coefficient initialization are excluded. CSV columns and speedup/efficiency calculations remain unchanged.

`correct` is `PASS` only if the signed Laplacian outputs agree within the existing absolute/relative tolerance (`1e-5`) and the edge maps match exactly. The reported `max_abs_error` and `rmse` describe the signed Laplacian responses.

Read the outputs in Python:

```python
import numpy as np
response = np.fromfile('response.bin', dtype='<f4').reshape(1024, 1024)
edges = np.fromfile('edges.bin', dtype='<f4').reshape(1024, 1024)
```

## Single-image script

The shell script compiles the benchmark if needed, runs the full pipeline, and creates a side-by-side preview:

```bash
./laplacian_openmp/test_one.sh 1024 image_01.bin
```

Outputs default to `results/laplacian/single/SIZE/`:

- `IMAGE_SIZE_metrics.csv`: timings, speedup, efficiency, and correctness.
- `IMAGE_SIZE_laplacian.bin`: signed float32 response.
- `IMAGE_SIZE_edges.bin`: float32 binary edge map.
- `IMAGE_SIZE_preview.png`: grayscale input beside the binary edge map.

To select another dataset or output directory:

```bash
DATASET_DIR=/path/to/data/processed RESULTS_DIR=/path/to/results \
  ./laplacian_openmp/test_one.sh 1024 image_01.bin
```

## Full dataset

Run all files in the 512, 1024, and 2048 folders:

```bash
./laplacian_openmp/run_full.sh
```

By default, results go to `results/laplacian/full/`:

- `raw_results.csv`: one Sequential and five OpenMP rows per image.
- `summary.csv`: medians by image size, implementation, and thread count.
- `previews/{512,1024,2048}/*_preview.png`: source grayscale beside the binary edge map.

Preview generation happens outside benchmark timing. Temporary signed-response and edge-map files are removed after each preview. To write CSV files only:

```bash
EXPORT_IMAGES=0 ./laplacian_openmp/run_full.sh
```

Alternative dataset and result paths can be passed to `run_full.sh`:

```bash
./laplacian_openmp/run_full.sh /path/to/data/processed /path/to/results
```

## Tests and dependencies

Run the numerical and integration tests from the repository root:

```bash
python3 -m unittest discover -s laplacian_openmp -p 'test_*.py' -v
```

Tests require Python 3, NumPy, and Pillow. The C implementation requires GCC, Make, and OpenMP. No additional runtime dependencies are used.
