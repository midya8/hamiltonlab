import json
import math
import random
import subprocess
import sys
import unittest
from hamiltonlab.finite_window import finite_window_audit,_window

class FiniteWindowTests(unittest.TestCase):
    def test_direct_covariance_double_sum(self):
        for rho in [-1.,-.999,-.8,-.1,0.,.2,.97,1.]:
            for n in [1,2,3,10,31,100]:
                direct=math.fsum(rho**abs(i-j) for i in range(n) for j in range(n))/(n*n)
                r=_window(rho,n)
                self.assertAlmostEqual(r['variance_of_mean_over_stationary_variance'],direct,13)
                self.assertAlmostEqual(r['signed_lag_formula_factor'],direct,13)
                self.assertGreaterEqual(r['variance_of_mean_over_stationary_variance'],0.)
    def test_independent_quarter_period(self):
        r=finite_window_audit([1.,100.],[1.,100.],math.pi/32,16,20)
        for k,c in zip([1.,100.],r['coordinates']):
            self.assertAlmostEqual(c['position']['variance_of_mean'],1/(20*k),14)
            self.assertAlmostEqual(c['squared_position']['variance_of_mean'],2/(20*k*k),14)
    def test_full_period_never_averages_magnitude(self):
        r=finite_window_audit([1.,4.],[1.,4.],math.pi/8,16,100)['coordinates'][0]
        self.assertEqual(r['position']['variance_of_mean'],1.)
        self.assertEqual(r['squared_position']['variance_of_mean'],2.)
        self.assertEqual(r['position']['variance_ratio_to_iid_mean'],100.)
    def test_half_period_even_odd(self):
        for n in [1,2,3,20,21]:
            r=finite_window_audit([1.,1.],[1.,1.],math.pi/8,8,n)['coordinates'][0]
            self.assertEqual(r['position']['variance_of_mean'],(n%2)/(n*n))
            self.assertEqual(r['squared_position']['variance_of_mean'],2.)
            self.assertEqual(r['squared_position']['lag_correlations'],[1.]*n)
    def test_all_audit_terms(self):
        r=_window(-.7,9)
        self.assertEqual(len(r['lag_correlations']),9);self.assertEqual(len(r['weighted_lag_terms']),8)
        self.assertEqual(r['lag_weights'],[1-j/9 for j in range(9)])
        for j in range(9):self.assertAlmostEqual(r['lag_correlations'][j],(-.7)**j,14)
        a=[sum((-.7)**j for j in range(t)) for t in range(1,10)]
        self.assertAlmostEqual(r['initial_state_sum_coefficient'],a[-1],14)
        self.assertAlmostEqual(r['innovation_coefficient_squares_sum'],sum(z*z for z in a[:-1]),14)
    def test_negative_correlation_position_not_square(self):
        r=finite_window_audit([1.,1.],[1.,1.],math.acos(-.8)/8,8,20)['coordinates'][0]
        self.assertLess(r['position']['variance_ratio_to_iid_mean'],1.)
        self.assertGreater(r['squared_position']['variance_ratio_to_iid_mean'],1.)
        for j,(p,q) in enumerate(zip(r['position']['lag_correlations'],r['squared_position']['lag_correlations'])):self.assertAlmostEqual(q,p*p,14)
    def test_seeded_stationary_exact_kernel(self):
        rho=-.6;n=8;rep=30000;rng=random.Random(15);sums=[];squares=[]
        for _ in range(rep):
            x=rng.gauss(0,1);chain=[x]
            for _ in range(n-1):x=rho*x+math.sqrt(1-rho*rho)*rng.gauss(0,1);chain.append(x)
            sums.append(sum(chain)/n);squares.append(sum(z*z for z in chain)/n)
        def variance(v):m=sum(v)/len(v);return sum((z-m)**2 for z in v)/(len(v)-1)
        self.assertLess(abs(variance(sums)-_window(rho,n)['variance_of_mean_over_stationary_variance']),.002)
        self.assertLess(abs(variance(squares)-2*_window(rho*rho,n)['variance_of_mean_over_stationary_variance']),.015)
    def test_limits_no_mutation_and_finite(self):
        for n in [0,10001,True,2.5]:
            with self.assertRaises(ValueError):finite_window_audit([1.,1.],[1.,1.],.1,2,n)
        k=[1.,2.];m=[2.,3.];r=finite_window_audit(k,m,.1,2,10000)
        self.assertEqual(k,[1.,2.]);self.assertEqual(m,[2.,3.]);json.dumps(r,allow_nan=False)
        with self.assertRaises((ValueError,OverflowError)):finite_window_audit([1e-300,1.],[1.,1.],.1,1,2)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','window','examples/window.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['draws'],20)
        r=subprocess.run([sys.executable,'-m','hamiltonlab','window','examples/harmonic.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2)
