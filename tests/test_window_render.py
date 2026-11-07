import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
spec=importlib.util.spec_from_file_location('render_windows',Path(__file__).resolve().parents[1]/'examples/render_windows.py');renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

class WindowRenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.v=renderer.curves()
    def test_full_ranges_and_counterexamples(self):
        for arm in self.v['arms']:self.assertEqual([p['draws'] for p in arm['points']],list(range(1,41)))
        flip=self.v['arms'][1]['points'];self.assertEqual(flip[19]['position_factor'],0.);self.assertEqual(flip[20]['position_factor'],1/441)
        r=self.v['arms'][3]['points'][19];self.assertEqual(r['position_factor'],.05);self.assertEqual(r['squared_position_factor'],1.)
    def test_svg_all_vertices_inside_shared_axes(self):
        root=ET.fromstring(renderer.render(self.v));paths=root.findall('{http://www.w3.org/2000/svg}polyline');self.assertEqual(len(paths),15)
        for idx,arm in enumerate(self.v['arms']):
            x=75+(idx%2)*550;y=165+(idx//2)*285
            for line in paths[idx*3:idx*3+3]:
                pts=[tuple(map(float,p.split(','))) for p in line.attrib['points'].split()];self.assertEqual(len(pts),40)
                self.assertEqual(pts[0][0],x);self.assertEqual(pts[-1][0],x+430)
                for px,py in pts:self.assertTrue(x<=px<=x+430);self.assertTrue(y-1e-6<=py<=y+160+1e-6)
    def test_actual_coordinates_match_data(self):
        root=ET.fromstring(renderer.render(self.v));paths=root.findall('{http://www.w3.org/2000/svg}polyline')
        for idx,arm in enumerate(self.v['arms']):
            y=165+(idx//2)*285
            for j,field in enumerate(['position_factor','squared_position_factor']):
                pts=[tuple(map(float,p.split(','))) for p in paths[idx*3+j].attrib['points'].split()]
                for p,(_,py) in zip(arm['points'],pts):self.assertAlmostEqual(py,y+160-160*p[field],5)
    def test_validation_missing_reordered_nonfinite(self):
        for patch in ['missing','order','nan','range','bool']:
            v=copy.deepcopy(self.v)
            if patch=='missing':v['arms'][0]['points'].pop()
            elif patch=='order':v['arms'][0]['points'][1]['draws']=1
            elif patch=='nan':v['arms'][0]['points'][1]['position_factor']=float('nan')
            elif patch=='range':v['arms'][0]['points'][1]['squared_position_factor']=1.1
            else:v['arms'][0]['points'][1]['draws']=True
            with self.assertRaises(ValueError):renderer.render(v)
    def test_xml_escape_and_no_mutation(self):
        v=copy.deepcopy(self.v);v['arms'][0]['title']='q < q & "test"';before=copy.deepcopy(v);s=renderer.render(v)
        self.assertIn('q &lt; q &amp; &quot;test&quot;',s);ET.fromstring(s);self.assertEqual(v,before)
    def test_deterministic_committed_asset(self):
        path=Path(__file__).resolve().parents[1]/'assets/window-audit.svg';self.assertEqual(path.read_text(),renderer.render(self.v));self.assertEqual(renderer.render(self.v),renderer.render(self.v))
    def test_cli_check_and_full_curve_json(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'curves.json';r=subprocess.run([sys.executable,'examples/render_windows.py','--check','--json',str(path)],capture_output=True,text=True)
            self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(path.read_text()),self.v)
