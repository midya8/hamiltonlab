import json
import math
import subprocess
import sys
import unittest
from hamiltonlab.harmonic import harmonic_audit
from hamiltonlab.quadratic import multiply

class HarmonicTests(unittest.TestCase):
    def test_closed_solution(self):
        r=harmonic_audit([4.,9.],[1.,4.],.2,3,[.4,-.1])
        for k,m,q,c in zip([4.,9.],[1.,4.],[.4,-.1],r['coordinates']):
            w=math.sqrt(k/m);phase=w*.6;co=math.cos(phase);si=math.sin(phase)
            self.assertAlmostEqual(c['frequency'],w);self.assertAlmostEqual(c['exact_trajectory_map'][0][0],co)
            self.assertAlmostEqual(c['exact_trajectory_map'][0][1],si/(m*w))
            self.assertAlmostEqual(c['exact_trajectory_map'][1][0],-m*w*si)
            self.assertAlmostEqual(c['exact_conditional_position_mean'],co*q)
            self.assertAlmostEqual(c['exact_conditional_position_variance'],si*si/k)
    def test_exact_covariance_energy_and_determinant(self):
        k=[.7,2.3];m=[2.,.4];q=[.3,-.2];p=[-.7,.4]
        r=harmonic_audit(k,m,.1,17,q)
        for i,c in enumerate(r['coordinates']):
            a,b=c['exact_trajectory_map'][0];cc,d=c['exact_trajectory_map'][1]
            qq=a*q[i]+b*p[i];pp=cc*q[i]+d*p[i]
            self.assertAlmostEqual(k[i]*qq*qq+pp*pp/m[i],k[i]*q[i]*q[i]+p[i]*p[i]/m[i],13)
            self.assertAlmostEqual(c['exact_determinant'],1.,14)
            self.assertAlmostEqual(c['exact_stationary_position_variance'],1/k[i],14)
            for row in range(2):
                for col in range(2):self.assertAlmostEqual(c['exact_standardized_stationary_joint_covariance'][row][col],float(row==col),14)
    def test_inverse_and_composition(self):
        def maps(e,n):return [c['exact_trajectory_map'] for c in harmonic_audit([2.,3.],[4.,.5],e,n)['coordinates']]
        for x,y,z,inv in zip(maps(.1,3),maps(.1,5),maps(.1,8),maps(-.1,3)):
            prod=multiply(x,y);eye=multiply(x,inv)
            for i in range(2):
                for j in range(2):self.assertAlmostEqual(prod[i][j],z[i][j],13);self.assertAlmostEqual(eye[i][j],float(i==j),13)
    def test_quarter_period_and_precision_mass(self):
        r=harmonic_audit([1.,100.],[1.,100.],math.pi/32,16)
        for k,c in zip([1.,100.],r['coordinates']):
            self.assertAlmostEqual(c['frequency'],1.,14);self.assertAlmostEqual(c['exact_stationary_lag_one_position_correlation'],0.,14)
            self.assertAlmostEqual(c['exact_conditional_position_variance'],1/k,14)
        identity=harmonic_audit([1.,100.],[1.,1.],math.pi/32,16)['coordinates'][1]
        self.assertAlmostEqual(identity['frequency'],10.);self.assertLess(identity['exact_conditional_position_variance'],1e-28)
        inverse=harmonic_audit([1.,100.],[1.,.01],math.pi/320,16)['coordinates'][1]
        self.assertAlmostEqual(inverse['frequency'],100.) # M is momentum covariance, not inverse mass
    def test_exact_resonance_and_large_esjd_trap(self):
        full=harmonic_audit([1.,1.],[1.,1.],math.pi/8,16)['coordinates'][0]
        half=harmonic_audit([1.,1.],[1.,1.],math.pi/8,8)['coordinates'][0]
        self.assertLess(full['exact_stationary_expected_squared_jump'],1e-28)
        self.assertAlmostEqual(half['exact_stationary_expected_squared_jump'],4.)
        self.assertAlmostEqual(half['exact_stationary_lag_one_position_correlation'],-1.)
        self.assertLess(half['exact_conditional_position_variance'],1e-28)
    def test_numerical_covariance_is_pre_correction(self):
        r=harmonic_audit([1.,1.],[1.,1.],math.sqrt(2),1)['coordinates'][0]
        self.assertAlmostEqual(r['leapfrog_stationary_position_variance_relative_error'],1.,13)
        self.assertAlmostEqual(r['leapfrog_standardized_stationary_joint_covariance'][0][0],2.,13)
        self.assertAlmostEqual(r['exact_standardized_stationary_joint_covariance'][0][0],1.,13)
    def test_fixed_duration_second_order(self):
        errors=[harmonic_audit([1.,4.],[1.,1.],1.3/n,n)['coordinates'][1]['standardized_map_error_frobenius'] for n in [8,16,32,64]]
        for a,b in zip(errors,errors[1:]):self.assertGreater(a/b,3.9);self.assertLess(a/b,4.1)
    def test_actual_error_matrix_and_no_mutation(self):
        k=[1.,4.];m=[2.,3.];q=[.2,-.3];r=harmonic_audit(k,m,.2,7,q)
        self.assertEqual(k,[1.,4.]);self.assertEqual(m,[2.,3.]);self.assertEqual(q,[.2,-.3])
        for c in r['coordinates']:
            error=c['standardized_map_error']
            self.assertAlmostEqual(c['standardized_map_error_frobenius'],math.sqrt(sum(z*z for row in error for z in row)),14)
            for i in range(2):
                for j in range(2):self.assertAlmostEqual(error[i][j],c['leapfrog_standardized_map'][i][j]-c['exact_standardized_rotation'][i][j],14)
    def test_invalid_bounds_and_nonfinite(self):
        for patch in [{'precision':[0.,1.]},{'mass':[True,1.]},{'position':[1.]},{'step_size':float('inf')},{'steps':1001},{'step_size':0.},{'steps':True},{'precision':[10001.,1.]},{'mass':[1.,float('nan')]}]:
            v=dict(precision=[1.,1.],mass=[1.,1.],step_size=.1,steps=3);v.update(patch)
            with self.assertRaises((ValueError,OverflowError)):harmonic_audit(**v)
        with self.assertRaises((ValueError,OverflowError)):harmonic_audit([1e4,1.],[1e-20,1.],1.,1)
    def test_cli_and_unknown_field(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','harmonic','examples/harmonic.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertAlmostEqual(v['duration'],math.pi/2)
        r=subprocess.run([sys.executable,'-m','hamiltonlab','harmonic','examples/chain.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,2);self.assertIn('requires',r.stderr)
