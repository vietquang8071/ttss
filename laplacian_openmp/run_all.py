#!/usr/bin/env python3
"""Benchmark all .bin images produced by preprocess.py; output raw + aggregate CSV."""
import argparse
import csv
import io
import statistics
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset/processed"))
    parser.add_argument("--exe", type=Path, default=Path("./laplacian_benchmark"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--export-images", action="store_true",
                        help="Export a side-by-side PNG preview per input image")
    args = parser.parse_args()

    exe = args.exe.resolve()
    if not exe.is_file():
        parser.error(f"Executable does not exist: {exe}. Compile first with 'make'.")
    tasks = []
    for size in (512, 1024, 2048):
        paths = sorted((args.dataset / str(size)).glob("*.bin"))
        if not paths:
            parser.error(f"No .bin files in {args.dataset / str(size)}")
        tasks.extend((size, path) for path in paths)

    args.out.mkdir(parents=True, exist_ok=True)
    visualizer = Path(__file__).resolve().with_name("visualize.py")
    if args.export_images:
        if not visualizer.is_file():
            parser.error(f"Visualization script not found: {visualizer}")
        try:
            import numpy  # noqa: F401
            from PIL import Image  # noqa: F401
        except ImportError:
            parser.error("--export-images requires NumPy and Pillow: pip install numpy pillow")

    all_rows = []
    # One temporary raw output file can be reused: PNG conversion happens
    # after C benchmarking, and the raw file is never needed for metrics.
    with tempfile.TemporaryDirectory(prefix="laplacian_preview_") as tmp:
        tmp_raw = Path(tmp) / "laplacian.bin"
        for i, (size, path) in enumerate(tasks, start=1):
            print(f"[{i}/{len(tasks)}] {path}", flush=True)
            command = [str(exe), str(path), str(size)]
            if args.export_images:
                command.extend(["--output", str(tmp_raw)])
            completed = subprocess.run(
                command, text=True, capture_output=True, check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"Benchmark failed for {path} (exit {completed.returncode}):\n"
                    f"{completed.stderr}\n{completed.stdout}"
                )
            rows = list(csv.DictReader(io.StringIO(completed.stdout)))
            if len(rows) != 8:
                raise ValueError(f"Unexpected benchmark CSV for {path}: {completed.stdout}")
            all_rows.extend(rows)

            if args.export_images:
                png = args.out / "previews" / str(size) / f"{path.stem}_preview.png"
                png.parent.mkdir(parents=True, exist_ok=True)
                render = subprocess.run(
                    [sys.executable, str(visualizer), "--input", str(path),
                     "--laplacian", str(tmp_raw), "--size", str(size),
                     "--out", str(png)],
                    text=True, capture_output=True, check=False,
                )
                if render.returncode != 0:
                    raise RuntimeError(
                        f"Preview generation failed for {path}:\n{render.stderr}\n{render.stdout}"
                    )
                print(f"  Preview: {png}", flush=True)

    fields = list(all_rows[0].keys())
    with (args.out / "raw_results.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(int(row["size"]), row["version"], int(row["threads"]))].append(row)

    summaries = []
    for (size, version, threads), rows in sorted(grouped.items()):
        summaries.append({
            "size": size,
            "version": version,
            "threads": threads,
            "images": len(rows),
            "median_execution_time_ms": statistics.median(float(r["median_ms"]) for r in rows),
            "median_speedup": statistics.median(float(r["speedup"]) for r in rows),
            "median_efficiency_percent": statistics.median(float(r["efficiency_percent"]) for r in rows),
            "max_abs_error_over_images": max(float(r["max_abs_error"]) for r in rows),
            "max_rmse_over_images": max(float(r["rmse"]) for r in rows),
            "all_correct": all(r["correct"] == "PASS" for r in rows),
        })

    with (args.out / "summary.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)

    print(f"Wrote {len(all_rows)} measurements to {args.out / 'raw_results.csv'}")
    print(f"Wrote {len(summaries)} aggregates to {args.out / 'summary.csv'}")
    if args.export_images:
        print(f"Wrote {len(tasks)} PNG previews to {args.out / 'previews'}")


if __name__ == "__main__":
    main()
