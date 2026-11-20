"""Integration tests for full dataset preview generation."""
import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
FAKE_RUNNER = '''#!/usr/bin/env python3
import numpy as np
import shutil
import sys
args = sys.argv[1:]
options = dict(zip(args[2::2], args[3::2]))
if '--output' in options:
    shutil.copyfile(args[0], options['--output'])
if '--edges-output' in options:
    size = int(args[1])
    edges = np.zeros((size, size), dtype='<f4')
    edges[1, 1] = 1.0
    edges[-2, -2] = 1.0
    edges.tofile(options['--edges-output'])
print('file,size,version,threads,median_ms,speedup,efficiency_percent,max_abs_error,rmse,correct')
for version,threads in [('sequential',0),('openmp',2),('openmp',4),('openmp',8),('openmp',12),('openmp',16)]:
    print(f'{args[0]},{args[1]},{version},{threads},1,1,100,0,0,PASS')
'''

class ExportPreviewsTest(unittest.TestCase):
    def run_case(self, export: bool):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'processed'
            for n in (512, 1024, 2048):
                directory = dataset / str(n)
                directory.mkdir(parents=True)
                with (directory / 'sample.bin').open('wb') as f:
                    f.truncate(n * n * 4)
            runner = root / 'fake_runner.py'
            runner.write_text(FAKE_RUNNER)
            runner.chmod(0o755)
            output = root / 'results'
            cmd = [sys.executable, str(HERE / 'run_all.py'),
                   '--dataset', str(dataset), '--exe', str(runner), '--out', str(output)]
            if export:
                cmd.append('--export-images')
            res = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, res.stderr)
            for name in ('raw_results.csv', 'summary.csv'):
                self.assertTrue((output / name).exists(), name)
            with (output / 'raw_results.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))), 18)
            for n in (512, 1024, 2048):
                png = output / 'previews' / str(n) / 'sample_preview.png'
                if export:
                    self.assertTrue(png.exists(), str(png))
                    with Image.open(png) as im:
                        self.assertEqual(im.size, (2 * n, n))
                        pixels = np.asarray(im.convert('L'))
                        edges = pixels[:, n:]
                        self.assertEqual(edges[1, 1], 255)
                        self.assertEqual(edges[-2, -2], 255)
                        self.assertEqual(edges[1, 2], 0)
                else:
                    self.assertFalse(png.exists())
            self.assertEqual(list(output.rglob('*_laplacian.bin')), [], 'temporary responses must be removed')
            self.assertEqual(list(output.rglob('*_edges.bin')), [], 'temporary edge maps must be removed')

    def test_export_png_for_all_images(self):
        self.run_case(export=True)

    def test_csv_only_by_default(self):
        self.run_case(export=False)

    def test_single_image_script_writes_response_edges_and_preview(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            dataset = root / 'processed'
            (dataset / '512').mkdir(parents=True)
            image = np.zeros((512, 512), dtype='<f4')
            image[:, 256:] = 100.0
            image.tofile(dataset / '512' / 'sample.bin')
            output = root / 'results'
            environment = os.environ.copy()
            environment['DATASET_DIR'] = str(dataset)
            environment['RESULTS_DIR'] = str(output)
            result = subprocess.run(
                ['bash', str(HERE / 'test_one.sh'), '512', 'sample.bin'],
                capture_output=True, text=True, env=environment, timeout=120,
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            single = output / 'single' / '512'
            response_path = single / 'sample_512_laplacian.bin'
            edge_path = single / 'sample_512_edges.bin'
            preview_path = single / 'sample_512_preview.png'
            for path in (response_path, edge_path, preview_path, single / 'sample_512_metrics.csv'):
                self.assertTrue(path.exists(), str(path))
            response = np.fromfile(response_path, dtype='<f4')
            edges = np.fromfile(edge_path, dtype='<f4')
            self.assertEqual(response.size, 512 * 512)
            self.assertEqual(edges.size, 512 * 512)
            self.assertTrue(np.isin(edges, (0.0, 1.0)).all())
            with (single / 'sample_512_metrics.csv').open() as file:
                rows = list(csv.DictReader(file))
            self.assertEqual(len(rows), 6)
            self.assertTrue(all(row['correct'] == 'PASS' for row in rows))
            with Image.open(preview_path) as image_preview:
                self.assertEqual(image_preview.size, (1024, 512))
                self.assertEqual(np.asarray(image_preview.convert('L'))[:, 512:].max(), 255)

if __name__ == '__main__':
    unittest.main()
