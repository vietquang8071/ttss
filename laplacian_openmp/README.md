# Laplacian Filtering — C/OpenMP

Implements and benchmarks the 3x3 four-neighbor Laplacian filter with zero padding:

```
 0  1  0
 1 -4  1
 0  1  0
```

Input `.bin`: no header, **little-endian IEEE-754 float32**, square `SIZE x SIZE`, row-major. These are the files created by the earlier `preprocess.py` script under `data/processed/{512,1024,2048}/`.

## Compile (Linux)

```bash
make
# or: gcc -O3 -std=c11 -Wall -Wextra -fopenmp laplacian_benchmark.c -o laplacian_benchmark -lm
```

## Single image

```bash
./laplacian_benchmark data/processed/1024/image_01.bin 1024
./laplacian_benchmark data/processed/1024/image_01.bin 1024 --output laplacian_output.bin
```

Output CSV: one Sequential row, then six OpenMP rows for thread counts 1/2/4/8/12/16. Each configuration is warmed up once and timed 10 times; reported timing is the median, and I/O is excluded. `--output` writes the *signed* raw Laplacian float32 response after benchmarking (it is the output of the OpenMP 16-thread run).

Read the saved output in Python:

```python
import numpy as np
x = np.fromfile('laplacian_output.bin', dtype='<f4').reshape(1024, 1024)
print(x.min(), x.max())
```

Do not clip the signed result to 0..255 before comparing correctness. For display, you can separately normalize absolute values or show a diverging colormap.

## Entire processed dataset (60 files)

From the project root (adjust paths as appropriate):

```bash
python3 path/to/run_all.py --dataset data/processed \
  --exe path/to/laplacian_benchmark --out results/laplacian/full --export-images
```

Outputs:

- `raw_results.csv`: measurements per file/algorithm/thread count.
- `summary.csv`: median over 20 images per size/variant/thread, with max correctness errors.
- `previews/{512,1024,2048}/*_preview.png`: side-by-side input and absolute Laplacian visualization (with `--export-images`).

## Important controls

Run all benchmarks on the same physical computer with minimal other CPU load. Try not to exceed the number of logical processors with OpenMP threads. This program uses `schedule(static)` and keeps input immutable, assigning independent rows of output to parallel threads. A reference Sequential output is compared to each OpenMP output using absolute and relative tolerance `1e-5`. Timing uses `omp_get_wtime()`.

The program checks binary file size and fails clearly if a `.bin` input is not the expected size (for example a 512x512 file erroneously passed as 1024). It cannot read JPG/PNG directly; run preprocessing first.

## Verify and visualize

```bash
python3 test_laplacian.py
# Requires numpy (and Pillow for visualization).
python3 visualize.py --input data/processed/1024/image_01.bin \
  --laplacian laplacian_output.bin --size 1024 --out laplacian_preview.png
```

`laplacian_preview.png` shows the input on the left and contrast-normalized
absolute Laplacian response on the right. This is *for visualization only*;
the signed float32 `.bin` remains the output used for correctness comparisons.

## Quick commands — Bash scripts

The two shell scripts can be run from **any working directory**. Their default
input directory is `../data/processed` for `run_full.sh` (or override it
with an argument). `test_one.sh` still defaults to `../data/processed`
unless you set `DATASET_DIR`. Both default to `../results/laplacian` for results.

```bash
chmod +x test_one.sh run_full.sh

# Test one 1024x1024 image, auto-selecting the first .bin file:
./test_one.sh

# Test a specific image / size:
./test_one.sh 512 image_01.bin
./test_one.sh 2048 image_05.bin

# Benchmark the entire dataset (512, 1024, 2048):
./run_full.sh
```

To use a dataset elsewhere, override the paths:

```bash
# Single image (uses DATASET_DIR and RESULTS_DIR environment variables):
DATASET_DIR=/path/to/data/processed RESULTS_DIR=/path/to/results ./test_one.sh 1024 image_01.bin

# Entire dataset (optional arguments):
./run_full.sh /path/to/data/processed /path/to/results
```

Single image outputs under `RESULTS_DIR/single/SIZE/`:

- `IMAGE_SIZE_metrics.csv`: median timings, speedup, efficiency, correctness
- `IMAGE_SIZE_laplacian.bin`: signed float32 output
- `IMAGE_SIZE_preview.png`: original grayscale + absolute Laplacian preview

Full dataset outputs under `RESULTS_DIR/full/`:

- `raw_results.csv`: 8 benchmark rows per image
- `summary.csv`: results aggregated by image size and thread count
- `previews/512/`, `previews/1024/`, `previews/2048/`: one PNG per input image
  showing grayscale input (left) and normalized absolute Laplacian (right).
  These previews are created **outside benchmark timing**; temporary raw
  `.bin` outputs are removed after PNG conversion to conserve disk space.

By default, `run_full.sh` exports PNG previews. To run CSV-only:

```bash
EXPORT_IMAGES=0 ./run_full.sh ../data/processed
```

`test_one.sh` and `run_full.sh` (when exporting images) need `python3`, NumPy,
Pillow, GCC, Make and OpenMP. You can install the Python requirements using
`python3 -m pip install numpy pillow`.
Neither script downloads files directly from Google Drive: download and
preprocess source images first.
