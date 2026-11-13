#!/usr/bin/env python3
"""Independent numerical and input-validation checks for Laplacian C/OpenMP."""
import csv
import io
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "laplacian_benchmark"


def reference_laplacian(image: np.ndarray) -> np.ndarray:
    """Independent reference with zero-padding and exact left-to-right float32 arithmetic."""
    padded = np.pad(image, 1, mode="constant", constant_values=0)
    n = image.shape[0]
    output = np.empty_like(image)
    for y in range(n):
        for x in range(n):
            output[y, x] = (
                padded[y + 1, x] + padded[y + 1, x + 2]
                + padded[y, x + 1] + padded[y + 2, x + 1]
                - np.float32(4) * image[y, x]
            )
    return output


class TestLaplacian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "-C", str(ROOT)], check=True, capture_output=True)

    def run_image(self, image: np.ndarray):
        n = image.shape[0]
        self.assertEqual(image.shape, (n, n))
        with tempfile.TemporaryDirectory() as temp:
            inp = Path(temp) / "input.bin"
            out = Path(temp) / "output.bin"
            np.asarray(image, dtype="<f4").tofile(inp)
            p = subprocess.run([str(EXE), str(inp), str(n), "--output", str(out)],
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            rows = list(csv.DictReader(io.StringIO(p.stdout)))
            self.assertEqual(len(rows), 8)
            self.assertEqual([int(row["threads"]) for row in rows], [0, 1, 2, 4, 8, 12, 16, 20])
            self.assertTrue(all(row["correct"] == "PASS" for row in rows))
            self.assertTrue(all(float(row["median_ms"]) >= 0 for row in rows))
            result = np.fromfile(out, dtype="<f4").reshape(n, n)
            return result

    def test_single_pixel_zero_padding(self):
        inp = np.array([[25]], dtype=np.float32)
        np.testing.assert_array_equal(self.run_image(inp), [[-100]])

    def test_impulse_and_signed_output(self):
        inp = np.zeros((3, 3), dtype=np.float32)
        inp[1, 1] = 100
        expected = np.array([[0, 100, 0], [100, -400, 100], [0, 100, 0]], dtype=np.float32)
        np.testing.assert_array_equal(self.run_image(inp), expected)

    def test_uniform_image_boundary(self):
        inp = np.full((5, 5), 6, dtype=np.float32)
        np.testing.assert_array_equal(self.run_image(inp), reference_laplacian(inp))

    def test_random_float32(self):
        rng = np.random.default_rng(42)
        inp = rng.uniform(0, 255, size=(7, 7)).astype(np.float32)
        np.testing.assert_allclose(self.run_image(inp), reference_laplacian(inp), atol=1e-5, rtol=1e-5)

    def test_file_size_error(self):
        with tempfile.TemporaryDirectory() as temp:
            inp = Path(temp) / "bad.bin"
            np.zeros((3, 3), dtype="<f4").tofile(inp)
            p = subprocess.run([str(EXE), str(inp), "512"], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("File size mismatch", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
