import copy
import itertools
import json
import math
import random
import subprocess
import sys
import unittest
from hamiltonlab.finite_window import mixture_window_audit,finite_window_audit,_window

def schedule(times):return [{'step_size':t/8,'steps':8} for t in times]

class MixtureWindowTests(unittest.TestCase):
    def test_singleton_matches_fixed_window(self):
        fixed=finite_window_audit([1.,3.],[2.,4.],.13,9,21)
        mix=mixture_window_audit([1.,3.],[2.,4.],[{'step_size':.13,'steps':9}],21)
        for a,b in zip(fixed['coordinates'],mix['coordinates']):
            self.assertEqual(a['position'],b['position']);self.assertEqual(a['squared_position'],b['squared_position'])
    def test_mean_zero_does_not_imply_magnitude_mixing(self):
        c=mixture_window_audit([1.,1.],[1.,1.],schedule([math.pi,2*math.pi]),20)['coordinates'][0]
        self.assertEqual(c['mean_cosine'],0.);self.assertEqual(c['mean_squared_cosine'],1.)
        self.assertEqual(c['square_of_mean_cosine'],0.)
        self.assertEqual(c['position']['variance_of_mean'],.05);self.assertEqual(c['squared_position']['variance_of_mean'],2.)
        self.assertEqual(c['squared_position']['lag_correlations'],[1.]*20)
    def test_nonresonant_symmetric_cosines(self):
        c=mixture_window_audit([1.,1.],[1.,1.],schedule([math.pi/3,2*math.pi/3]),20)['coordinates'][0]
        self.assertAlmostEqual(c['mean_cosine'],0.,14);self.assertAlmostEqual(c['mean_squared_cosine'],.25,14)
        self.assertAlmostEqual(c['position']['variance_of_mean'],.05,14)
        self.assertAlmostEqual(c['squared_position']['variance_of_mean'],2*_window(.25,20)['variance_of_mean_over_stationary_variance'],14)
        self.assertGreater(c['squared_position']['variance_ratio_to_iid_mean'],1.6)
    def test_exhaustive_path_products(self):
        rhos=[-.8,.2,.7];n=7
        c=mixture_window_audit([1.,1.],[1.,1.],schedule([math.acos(r) for r in rhos]),n)['coordinates'][0]
        for lag in range(1,n):
            products=[math.prod(p) for p in itertools.product(rhos,repeat=lag)]
            self.assertAlmostEqual(c['position']['lag_correlations'][lag],sum(products)/len(products),13)
            self.assertAlmostEqual(c['squared_position']['lag_correlations'][lag],sum(p*p for p in products)/len(products),13)
    def test_duplicate_multiplicity_and_order(self):
        ts=schedule([math.pi,math.pi,2*math.pi]);r=mixture_window_audit([1.,1.],[1.,1.],ts,17)
        c=r['coordinates'][0];self.assertAlmostEqual(c['mean_cosine'],-1/3);self.assertEqual(c['mean_squared_cosine'],1.)
        rev=mixture_window_audit([1.,1.],[1.,1.],list(reversed(ts)),17)['coordinates'][0]
        self.assertEqual(c['position'],rev['position']);self.assertEqual(c['squared_position'],rev['squared_position'])
        self.assertAlmostEqual(c['cosine_variance'],c['mean_squared_cosine']-c['square_of_mean_cosine'],14)
    def test_no_mutation_and_return_alias(self):
        ts=schedule([1.,2.]);before=copy.deepcopy(ts);k=[1.,2.];m=[2.,3.]
        r=mixture_window_audit(k,m,ts,10);self.assertEqual(ts,before);r['trajectories'][0]['steps']=99
        self.assertEqual(ts,before);self.assertEqual(k,[1.,2.]);self.assertEqual(m,[2.,3.])
    def test_seeded_exact_kernel(self):
        times=[math.pi/3,2*math.pi/3];rng=random.Random(42);n=10;rep=25000;means=[];squares=[]
        for _ in range(rep):
            q=rng.gauss(0,1);xs=[q]
            for _ in range(n-1):
                t=times[rng.randrange(2)];q=math.cos(t)*q+math.sin(t)*rng.gauss(0,1);xs.append(q)
            means.append(sum(xs)/n);squares.append(sum(q*q for q in xs)/n)
        def variance(xs):mean=sum(xs)/len(xs);return sum((x-mean)**2 for x in xs)/(len(xs)-1)
        c=mixture_window_audit([1.,1.],[1.,1.],schedule(times),n)['coordinates'][0]
        self.assertLess(abs(variance(means)-c['position']['variance_of_mean']),.003)
        self.assertLess(abs(variance(squares)-c['squared_position']['variance_of_mean']),.02)
    def test_bad_inputs_and_max_schedule(self):
        for ts in [[],{},[{}],[{'step_size':.1,'steps':1,'weight':1}],schedule([1.])*65,[{'step_size':0.,'steps':8}],[{'step_size':.1,'steps':True}]]:
            with self.assertRaises((ValueError,TypeError)):mixture_window_audit([1.,1.],[1.,1.],ts,20)
        for n in [0,10001,True]:
            with self.assertRaises(ValueError):mixture_window_audit([1.,1.],[1.,1.],schedule([1.]),n)
        r=mixture_window_audit([1.,1.],[1.,1.],schedule([1.])*64,10000);json.dumps(r,allow_nan=False)
        with self.assertRaises((ValueError,OverflowError)):mixture_window_audit([1e-300,1.],[1.,1.],schedule([1.]),2)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','window-mixture','examples/window_mixture.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['draws'],20)
        r=subprocess.run([sys.executable,'-m','hamiltonlab','window-mixture','examples/window.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
