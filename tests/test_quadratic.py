import json
import math
import random
import subprocess
import sys
import unittest
from hamiltonlab.quadratic import quadratic_map
from hamiltonlab.core import leapfrog,sample
from hamiltonlab.targets import standard

class QuadraticTests(unittest.TestCase):
    def test_actual_integrator_map(self):
        k=[.7,2.3];m=[2.,.4];q=[.3,-.2];p=[-.7,.4]
        def target(q):return sum(k[i]*q[i]**2/2 for i in range(2)),[k[i]*q[i] for i in range(2)]
        for n in [1,3,17]:
            audit=quadratic_map(k,m,.1,n,q);r=leapfrog(target,q,p,.1,n,m)
            for i,c in enumerate(audit['coordinates']):
                a,b=c['trajectory_map'];self.assertAlmostEqual(r['position'][i],a[0]*q[i]+a[1]*p[i],12);self.assertAlmostEqual(r['momentum'][i],b[0]*q[i]+b[1]*p[i],12)
                self.assertAlmostEqual(c['determinant'],1.,12)
    def test_quarter_and_full_period(self):
        a=quadratic_map([1.,1.],[1.,1.],math.sqrt(2),1)['coordinates'][0]
        b=quadratic_map([1.,1.],[1.,1.],math.sqrt(2),4)['coordinates'][0]
        self.assertAlmostEqual(a['trajectory_map'][0][0],0.,14);self.assertAlmostEqual(a['conditional_position_variance'],2.,14)
        self.assertAlmostEqual(a['stationary_input_proposal_variance'],2.,14) # not target variance, accept test still matters
        for i in range(2):
            for j in range(2):self.assertAlmostEqual(b['trajectory_map'][i][j],float(i==j),14)
        self.assertLess(b['conditional_position_variance'],1e-28);self.assertLess(b['stationary_input_expected_squared_jump'],1e-28)
    def test_energy_defect_matches_actual(self):
        q=[.3,-.2];p=[-.7,.4];a=quadratic_map([1.,1.],[1.,1.],.3,7);r=leapfrog(standard,q,p,.3,7)
        value=0.
        for i,c in enumerate(a['coordinates']):
            e=c['hamiltonian_quadratic_defect'];value+=.5*(q[i]**2*e[0][0]+2*q[i]*p[i]*e[0][1]+p[i]**2*e[1][1])
        self.assertAlmostEqual(value,r['energy_error'],13)
    def test_inverse_and_negative_time(self):
        a=quadratic_map([1.,2.],[3.,4.],.2,9);b=quadratic_map([1.,2.],[3.,4.],-.2,9)
        from hamiltonlab.quadratic import multiply
        for x,y in zip(a['coordinates'],b['coordinates']):
            z=multiply(x['trajectory_map'],y['trajectory_map'])
            for i in range(2):
                for j in range(2):self.assertAlmostEqual(z[i][j],float(i==j),13)
    def test_conditional_moments_seeded(self):
        a=quadratic_map([1.,1.],[2.,3.],.3,3,[.7,-.2]);rng=random.Random(8);n=20000
        for i,c in enumerate(a['coordinates']):
            aa,bb=c['trajectory_map'][0];q=a['position'][i];x=[aa*q+bb*rng.gauss(0,math.sqrt(a['mass'][i])) for _ in range(n)]
            mean=sum(x)/n;var=sum((z-mean)**2 for z in x)/(n-1)
            self.assertLess(abs(mean-c['conditional_position_mean']),.025);self.assertLess(abs(var-c['conditional_position_variance']),.025)
    def test_mass_precision_scale_invariance(self):
        a=quadratic_map([1.,2.],[1.,2.],.2,7);b=quadratic_map([3.,6.],[3.,6.],.2,7)
        for x,y in zip(a['coordinates'],b['coordinates']):
            self.assertAlmostEqual(x['trajectory_map'][0][0],y['trajectory_map'][0][0],13)
            self.assertAlmostEqual(x['trajectory_map'][0][1]/3,y['trajectory_map'][0][1],13)
    def test_stability_is_not_exploration(self):
        a=quadratic_map([1.,1.],[1.,1.],math.sqrt(2),4)
        self.assertTrue(a['coordinates'][0]['strictly_inside_linear_stability_region'])
        r=sample(standard,[2.,-1.],draws=100,warmup=0,step_size=math.sqrt(2),steps=4,seed=9)
        self.assertGreater(r['retained_acceptance_rate'],.99)
        self.assertLess(max(abs(q[0]-2) for q in r['samples']),1e-11)
        self.assertFalse(quadratic_map([1.,1.],[1.,1.],2.,1)['coordinates'][0]['strictly_inside_linear_stability_region'])
    def test_invalid(self):
        for kwargs in [{'precision':[0,1]},{'mass':[1,float('nan')]},{'step_size':0},{'steps':True},{'steps':1001},{'precision':[10001,1]}]:
            v=dict(precision=[1,1],mass=[1,1],step_size=.1,steps=3);v.update(kwargs)
            with self.assertRaises(ValueError):quadratic_map(**v)
        with self.assertRaises((ValueError,OverflowError)):quadratic_map([1e4,1e4],[.001,.001],10.,1000)
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','quadratic','examples/quadratic.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['steps'],4)
