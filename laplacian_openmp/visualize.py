#!/usr/bin/env python3
"""Create a side-by-side PNG from an input image and signed Laplacian .bin output."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Preprocessed float32 grayscale input .bin")
    p.add_argument("--laplacian", type=Path, required=True, help="Signed float32 Laplacian output .bin")
    p.add_argument("--size", type=int, required=True, help="Width = height of square images")
    p.add_argument("--out", type=Path, default=Path("laplacian_preview.png"))
    args = p.parse_args()

    n = args.size
    if n <= 0:
        p.error("--size must be positive")
    expected = n * n
    original = np.fromfile(args.input, dtype="<f4")
    response = np.fromfile(args.laplacian, dtype="<f4")
    if original.size != expected or response.size != expected:
        p.error(f"Expected {expected} float32 pixels in each file, found {original.size} and {response.size}")

    original = np.clip(original.reshape(n, n), 0, 255).astype(np.uint8)
    magnitude = np.abs(response.reshape(n, n))
    # Visual only: keep the signed .bin as-is for correctness tests.
    max_magnitude = float(magnitude.max())
    if max_magnitude > 0:
        magnitude = magnitude * (255.0 / max_magnitude)
    edge_preview = np.clip(magnitude, 0, 255).astype(np.uint8)
    canvas = Image.new("L", (2 * n, n))
    canvas.paste(Image.fromarray(original, mode="L"), (0, 0))
    canvas.paste(Image.fromarray(edge_preview, mode="L"), (n, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved {args.out} (left: grayscale input; right: abs(Laplacian) visualization)")


if __name__ == "__main__":
    main()
