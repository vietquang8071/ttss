#!/usr/bin/env python3
"""Prepare grayscale float32 image datasets for C++/OpenMP convolution benchmarks."""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZES = (512, 1024, 2048)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def oriented_image(path: Path) -> Image.Image:
    """Read an image and apply its EXIF orientation if present."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image)


def preprocess(input_dir: Path, output_dir: Path, expected: int, preview: bool) -> None:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    image_paths = sorted(
        p for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    )
    if not image_paths:
        raise ValueError(f"No supported images found in {input_dir}")
    if expected > 0 and len(image_paths) != expected:
        raise ValueError(f"Expected {expected} images, found {len(image_paths)}")

    image_ids = set()
    for path in image_paths:
        image_id = path.stem.casefold()
        if image_id in image_ids:
            raise ValueError(f"Multiple input images share the same output name: {path.stem}")
        image_ids.add(image_id)

    output_names = {f"{path.stem}.bin" for path in image_paths}
    for size in IMAGE_SIZES:
        folder = output_dir / str(size)
        extra = {path.name for path in folder.glob("*.bin")} - output_names
        if extra:
            raise ValueError(f"{folder} contains other .bin files. Choose a new output directory.")

    # Validate every image before writing any output files.
    invalid = []
    for path in image_paths:
        with oriented_image(path) as image:
            w, h = image.size
        if w < 2048 or h < 2048:
            invalid.append(f"{path.name}: {w}x{h}")
    if invalid:
        raise ValueError("Images smaller than 2048x2048:\n" + "\n".join(invalid))

    for size in IMAGE_SIZES:
        (output_dir / str(size)).mkdir(parents=True, exist_ok=True)
    if preview:
        (output_dir / "preview").mkdir(parents=True, exist_ok=True)

    metadata = []
    for index, path in enumerate(image_paths, start=1):
        with oriented_image(path) as original:
            w, h = original.size
            gray = original.convert("L")  # 8-bit luminance in [0, 255]

        side = min(w, h)
        left = (w - side) // 2
        top = (h - side) // 2
        square = gray.crop((left, top, left + side, top + side))
        image_id = path.stem

        for size in IMAGE_SIZES:
            resized = square.resize((size, size), resample=Image.Resampling.LANCZOS)
            pixels = np.asarray(resized, dtype="<f4")  # little-endian float32
            destination = output_dir / str(size) / f"{image_id}.bin"
            pixels.tofile(destination)  # raw HxW, row-major, no header

            metadata.append({
                "image_id": image_id,
                "original_file": path.name,
                "original_width": w,
                "original_height": h,
                "crop_left": left,
                "crop_top": top,
                "crop_size": side,
                "output_width": size,
                "output_height": size,
                "dtype": "float32_le",
                "pixel_range": "0..255",
                "relative_file": destination.relative_to(output_dir).as_posix(),
            })

            if preview and size == 512:
                resized.save(output_dir / "preview" / f"{image_id}.png")

        print(f"[{index:02d}/{len(image_paths):02d}] {path.name} ({w}x{h}) -> {image_id}")

    fields = list(metadata[0])
    with (output_dir / "metadata.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metadata)

    print(f"\nDone: {len(image_paths)} originals -> {len(metadata)} .bin files")
    print(f"Output: {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/home/vitquay1708/Study_Space/Parallel_Computing/data/raw"))
    parser.add_argument("--output", type=Path, default=Path("/home/vitquay1708/Study_Space/Parallel_Computing/data/processed"))
    parser.add_argument("--expected", type=int, default=20,
                        help="Expected number of originals; 0 disables this check")
    parser.add_argument("--preview", action="store_true",
                        help="Also save 512x512 grayscale PNG previews")
    args = parser.parse_args()
    preprocess(args.input, args.output, args.expected, args.preview)


if __name__ == "__main__":
    main()
