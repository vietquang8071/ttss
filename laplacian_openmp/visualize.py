#!/usr/bin/env python3
"""Create a side-by-side PNG from an input image and binary edge-map .bin output."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="Preprocessed float32 grayscale input .bin")
    p.add_argument("--edges", type=Path, required=True, help="Float32 edge map containing only 0.0 or 1.0")
    p.add_argument("--size", type=int, required=True, help="Width = height of square images")
    p.add_argument("--out", type=Path, default=Path("edge_preview.png"))
    args = p.parse_args()

    n = args.size
    if n <= 0:
        p.error("--size must be positive")
    expected = n * n
    original = np.fromfile(args.input, dtype="<f4")
    edges = np.fromfile(args.edges, dtype="<f4")
    if original.size != expected or edges.size != expected:
        p.error(f"Expected {expected} float32 pixels in each file, found {original.size} and {edges.size}")

    original = np.clip(original.reshape(n, n), 0, 255).astype(np.uint8)
    edges = edges.reshape(n, n)
    if not np.isin(edges, (0.0, 1.0)).all():
        p.error("Edge map must contain only float32 values 0.0 and 1.0")
    edge_preview = (edges * 255.0).astype(np.uint8)
    canvas = Image.new("L", (2 * n, n))
    canvas.paste(Image.fromarray(original, mode="L"), (0, 0))
    canvas.paste(Image.fromarray(edge_preview, mode="L"), (n, 0))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.out)
    print(f"Saved {args.out} (left: grayscale input; right: binary zero-crossing edge map)")


if __name__ == "__main__":
    main()
