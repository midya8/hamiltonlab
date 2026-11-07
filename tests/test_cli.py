import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

class CliTests(unittest.TestCase):
    def test_sample(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','sample','examples/chain.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertEqual(len(v['samples']),1000);self.assertEqual(v['potential_gradient_callback_calls'],13201)
    def test_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'bad.json';p.write_text('{"target":"nuts"}')
            r=subprocess.run([sys.executable,'-m','hamiltonlab','sample',str(p)],capture_output=True,text=True)
            self.assertEqual(r.returncode,2);self.assertNotIn('Traceback',r.stderr)
