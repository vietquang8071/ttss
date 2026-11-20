#!/usr/bin/env python3
"""Independent numerical and input-validation checks for the C/OpenMP LoG pipeline."""
import csv
import io
import math
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
EXE = ROOT / "laplacian_benchmark"


def reference_pipeline(image: np.ndarray, sigma: float = 1.0,
                      threshold: float = 10.0):
    """Independent float32 reference for separable Gaussian, Laplacian and crossings."""
    n = image.shape[0]
    radius = 2
    weights64 = np.array(
        [math.exp(-0.5 * (offset / sigma) * (offset / sigma))
         for offset in range(-radius, radius + 1)],
        dtype=np.float64,
    )
    weights = (weights64 / weights64.sum()).astype(np.float32)
    source = np.asarray(image, dtype=np.float32)
    horizontal = np.empty_like(source)
    smoothed = np.empty_like(source)

    for y in range(n):
        for x in range(n):
            value = np.float32(0)
            for k, weight in enumerate(weights):
                sx = min(max(x + k - radius, 0), n - 1)
                value = np.float32(value + np.float32(source[y, sx] * weight))
            horizontal[y, x] = value

    for y in range(n):
        for x in range(n):
            value = np.float32(0)
            for k, weight in enumerate(weights):
                sy = min(max(y + k - radius, 0), n - 1)
                value = np.float32(value + np.float32(horizontal[sy, x] * weight))
            smoothed[y, x] = value

    laplacian = np.empty_like(source)
    for y in range(n):
        for x in range(n):
            left = smoothed[y, x - 1] if x > 0 else np.float32(0)
            right = smoothed[y, x + 1] if x + 1 < n else np.float32(0)
            up = smoothed[y - 1, x] if y > 0 else np.float32(0)
            down = smoothed[y + 1, x] if y + 1 < n else np.float32(0)
            value = np.float32(left + right)
            value = np.float32(value + up)
            value = np.float32(value + down)
            value = np.float32(value - np.float32(4) * smoothed[y, x])
            laplacian[y, x] = value

    edges = np.zeros_like(source)
    for y in range(1, n - 1):
        for x in range(1, n - 1):
            center = float(laplacian[y, x])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if not (0 < ny < n - 1 and 0 < nx < n - 1):
                        continue
                    neighbor = float(laplacian[ny, nx])
                    if center * neighbor < 0 and abs(center) + abs(neighbor) >= threshold:
                        edges[y, x] = np.float32(1)
                        break
                if edges[y, x]:
                    break
    return laplacian, edges


class TestLaplacian(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        subprocess.run(["make", "-C", str(ROOT)], check=True, capture_output=True)

    def run_image(self, image: np.ndarray, *, sigma=None, threshold=None):
        n = image.shape[0]
        self.assertEqual(image.shape, (n, n))
        with tempfile.TemporaryDirectory() as temp:
            inp = Path(temp) / "input.bin"
            out = Path(temp) / "output.bin"
            edge_out = Path(temp) / "edges.bin"
            np.asarray(image, dtype="<f4").tofile(inp)
            command = [str(EXE), str(inp), str(n), "--output", str(out),
                       "--edges-output", str(edge_out)]
            if sigma is not None:
                command.extend(["--sigma", str(sigma)])
            if threshold is not None:
                command.extend(["--threshold", str(threshold)])
            p = subprocess.run(command,
                               capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
            rows = list(csv.DictReader(io.StringIO(p.stdout)))
            self.assertEqual(len(rows), 6)
            self.assertEqual([int(row["threads"]) for row in rows], [0, 2, 4, 8, 12, 16])
            self.assertTrue(all(row["correct"] == "PASS" for row in rows))
            self.assertTrue(all(float(row["median_ms"]) >= 0 for row in rows))
            result = np.fromfile(out, dtype="<f4").reshape(n, n)
            edges = np.fromfile(edge_out, dtype="<f4").reshape(n, n)
            self.assertTrue(np.isin(edges, (0.0, 1.0)).all())
            return result, edges

    def test_gaussian_laplacian_matches_reference_for_default_and_custom_sigma(self):
        inp = np.zeros((5, 5), dtype=np.float32)
        inp[2, 2] = 100
        for sigma in (None, 0.75):
            actual, edges = self.run_image(inp, sigma=sigma)
            expected, expected_edges = reference_pipeline(
                inp, sigma=1.0 if sigma is None else sigma,
            )
            np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(edges, expected_edges)

    def test_very_small_positive_sigma_produces_finite_outputs(self):
        inp = np.zeros((5, 5), dtype=np.float32)
        inp[2, 2] = 100
        actual, edges = self.run_image(inp, sigma=1e-200, threshold=0)
        expected, expected_edges = reference_pipeline(inp, sigma=1e-200, threshold=0)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        np.testing.assert_array_equal(edges, expected_edges)

    def test_step_edge_marks_both_interior_pixels_and_excludes_border(self):
        inp = np.tile(np.array([0, 0, 0, 100, 100, 100, 100], dtype=np.float32), (7, 1))
        _, edges = self.run_image(inp, threshold=0)
        expected = np.zeros((7, 7), dtype=np.float32)
        expected[1:-1, 2:4] = 1
        np.testing.assert_array_equal(edges, expected)
        self.assertFalse(edges[0, :].any())
        self.assertFalse(edges[-1, :].any())
        self.assertFalse(edges[:, 0].any())
        self.assertFalse(edges[:, -1].any())

    def test_diagonal_only_zero_crossings_are_detected(self):
        inp = np.zeros((7, 7), dtype=np.float32)
        inp[3, 3] = 100
        _, edges = self.run_image(inp, threshold=0)
        _, expected = reference_pipeline(inp, threshold=0)
        np.testing.assert_array_equal(edges, expected)
        for y, x in ((1, 1), (1, 5), (5, 1), (5, 5)):
            self.assertEqual(edges[y, x], 1.0)

    def test_zero_crossing_threshold_includes_equality_and_suppresses_weaker_pairs(self):
        inp = np.tile(np.array([0, 0, 0, 100, 100, 100, 100], dtype=np.float32), (7, 1))
        response, _ = reference_pipeline(inp)
        threshold = abs(float(response[3, 2])) + abs(float(response[3, 3]))
        _, at_threshold = self.run_image(inp, threshold=threshold)
        _, above_threshold = self.run_image(inp, threshold=np.nextafter(threshold, math.inf))
        self.assertTrue(at_threshold[3, 2] == 1 and at_threshold[3, 3] == 1)
        self.assertFalse(above_threshold.any())

    def test_size_one_and_constant_image_have_no_edges(self):
        _, single_edges = self.run_image(np.array([[25]], dtype=np.float32))
        np.testing.assert_array_equal(single_edges, [[0]])
        _, constant_edges = self.run_image(np.full((5, 5), 6, dtype=np.float32))
        np.testing.assert_array_equal(constant_edges, np.zeros((5, 5), dtype=np.float32))

    def test_invalid_sigma_and_threshold_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            inp = Path(temp) / "input.bin"
            np.zeros((3, 3), dtype="<f4").tofile(inp)
            for option, value in (("--sigma", "0"), ("--sigma", "nan"),
                                  ("--threshold", "-1"), ("--threshold", "inf")):
                result = subprocess.run(
                    [str(EXE), str(inp), "3", option, value],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0, (option, value))
                self.assertIn(option, result.stderr)

    def test_conflicting_output_destinations_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            inp = root / "input.bin"
            np.zeros((3, 3), dtype="<f4").tofile(inp)

            same_path = root / "same.bin"
            result = subprocess.run(
                [str(EXE), str(inp), "3", "--output", str(same_path),
                 "--edges-output", str(same_path)],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("different files", result.stderr)
            self.assertEqual(result.stdout, "")
            self.assertFalse(same_path.exists())

            target = root / "existing.bin"
            sentinel = b"leave this output untouched"
            target.write_bytes(sentinel)
            hard_link = root / "hard-link.bin"
            os.link(target, hard_link)
            symbolic_link = root / "symbolic-link.bin"
            symbolic_link.symlink_to(target)
            for alias in (hard_link, symbolic_link):
                result = subprocess.run(
                    [str(EXE), str(inp), "3", "--output", str(target),
                     "--edges-output", str(alias)],
                    capture_output=True, text=True,
                )
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("different files", result.stderr)
                self.assertEqual(target.read_bytes(), sentinel)

    def test_file_size_error(self):
        with tempfile.TemporaryDirectory() as temp:
            inp = Path(temp) / "bad.bin"
            np.zeros((3, 3), dtype="<f4").tofile(inp)
            p = subprocess.run([str(EXE), str(inp), "512"], capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("File size mismatch", p.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
