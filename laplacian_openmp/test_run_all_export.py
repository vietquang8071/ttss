"""Integration tests for full dataset preview generation."""
import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
FAKE_RUNNER = '''#!/usr/bin/env python3
import shutil
import sys
args = sys.argv[1:]
if len(args) == 4 and args[2] == '--output':
    shutil.copyfile(args[0], args[3])
print('file,size,version,threads,median_ms,speedup,efficiency_percent,max_abs_error,rmse,correct')
for version,threads in [('sequential',0),('openmp',1),('openmp',2),('openmp',4),('openmp',8),('openmp',12),('openmp',16)]:
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
                self.assertEqual(len(list(csv.DictReader(f))), 24)
            for n in (512, 1024, 2048):
                png = output / 'previews' / str(n) / 'sample_preview.png'
                if export:
                    self.assertTrue(png.exists(), str(png))
                    with Image.open(png) as im:
                        self.assertEqual(im.size, (2 * n, n))
                else:
                    self.assertFalse(png.exists())
            self.assertEqual(list(output.rglob('*_laplacian.bin')), [], 'temp outputs must be removed')

    def test_export_png_for_all_images(self):
        self.run_case(export=True)

    def test_csv_only_by_default(self):
        self.run_case(export=False)

if __name__ == '__main__':
    unittest.main()
