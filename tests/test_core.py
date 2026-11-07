import copy
import math
import unittest
from hamiltonlab.core import leapfrog,sample,accept
from hamiltonlab.targets import correlated,mixture


def standard(q):return sum(x*x for x in q)/2,q[:]

class CoreTests(unittest.TestCase):
    def test_target_gradients(self):
        for target in [correlated,mixture]:
            for q in [[.2,-.3],[-6.,1.],[5.,-2.]]:
                u,g=target(q)
                for k in range(2):
                    a=q[:];b=q[:];a[k]+=1e-5;b[k]-=1e-5
                    self.assertAlmostEqual(g[k],(target(a)[0]-target(b)[0])/2e-5,7)
    def test_closed_one_step(self):
        r=leapfrog(standard,[1.,0.],[0.,0.],.1,1)
        self.assertEqual(r['position'],[.995,0.]);self.assertAlmostEqual(r['momentum'][0],-.09975,14)
        self.assertEqual(r['gradient_evaluations'],2)
    def test_reversibility(self):
        for target in [standard,correlated,mixture]:
            a=leapfrog(target,[.3,.1],[.2,-.3],.03,17,[2.,.5])
            b=leapfrog(target,a['position'],[-x for x in a['momentum']],.03,17,[2.,.5])
            for x,y in zip(b['position'],[.3,.1]):self.assertAlmostEqual(x,y,12)
            for x,y in zip(b['momentum'],[-.2,.3]):self.assertAlmostEqual(x,y,12)
    def test_volume_finite_difference(self):
        q=[.2,.1,.3,-.2];h=1e-5;columns=[]
        for k in range(4):
            a=q[:];b=q[:];a[k]+=h;b[k]-=h
            def f(z):
                r=leapfrog(correlated,z[:2],z[2:],.1,5);return r['position']+r['momentum']
            columns.append([(x-y)/(2*h) for x,y in zip(f(a),f(b))])
        mat=[[columns[j][i] for j in range(4)] for i in range(4)];det=1.
        for k in range(4):
            pivot=mat[k][k];det*=pivot
            for i in range(k+1,4):
                ratio=mat[i][k]/pivot
                for j in range(k+1,4):mat[i][j]-=ratio*mat[k][j]
        self.assertAlmostEqual(det,1.,8)
    def test_energy_second_order(self):
        errors=[abs(leapfrog(standard,[1.,.3],[.4,-.2],h,round(1/h))['energy_error']) for h in [.2,.1,.05]]
        self.assertLess(errors[1],errors[0]/3);self.assertLess(errors[2],errors[1]/3)
    def test_acceptance_log_domain(self):
        self.assertTrue(accept(-.1,-1000000.));self.assertFalse(accept(-.1,1000000.));self.assertFalse(accept(-2.,2.));self.assertTrue(accept(-2.1,2.))
    def test_mass_and_negative_time(self):
        r=leapfrog(standard,[1.,2.],[.3,.2],.05,7,[2.,3.]);s=leapfrog(standard,r['position'],r['momentum'],-.05,7,[2.,3.])
        for x,y in zip(s['position']+s['momentum'],[1.,2.,.3,.2]):self.assertAlmostEqual(x,y,13)
    def test_no_mutation_and_determinism(self):
        q=[.2,.3];old=q[:];a=sample(standard,q,draws=20,warmup=5,seed=8);b=sample(standard,q,draws=20,warmup=5,seed=8)
        self.assertEqual(a,b);self.assertEqual(q,old);self.assertEqual(len(a['samples']),20);self.assertEqual(len(a['transitions']),25)
    def test_repeat_rejections_and_work(self):
        r=sample(standard,[1.,1.],draws=20,warmup=2,step_size=5.,steps=10,seed=5)
        self.assertEqual(r['retained_acceptance_rate'],0.);self.assertEqual(r['samples'],[[1.,1.]]*20)
        self.assertEqual(r['potential_gradient_callback_calls'],1+22*11)
        for t in r['transitions']:self.assertEqual(t['state'],t['before'])
    def test_invalid_trajectory_rejected_and_counted(self):
        def bounded(q):
            if abs(q[0])>.01:raise ValueError('outside numerical domain')
            return standard(q)
        r=sample(bounded,[0.,0.],draws=3,warmup=0,step_size=1.,steps=3,seed=2)
        self.assertEqual(r['invalid_trajectories'],3);self.assertEqual(r['samples'],[[0.,0.]]*3)
        self.assertEqual(r['potential_gradient_callback_calls'],7)
        self.assertTrue(all(t['energy_error'] is None for t in r['transitions']))
    def test_random_walk_work(self):
        r=sample(standard,[0.,0.],method='random_walk',draws=30,warmup=10,seed=4)
        self.assertEqual(r['potential_gradient_callback_calls'],41);self.assertIsNone(r['steps'])
    def test_gaussian_reference_moments(self):
        r=sample(standard,[0.,0.],draws=3000,warmup=200,step_size=.2,steps=8,seed=17)
        for k in range(2):
            mu=sum(q[k] for q in r['samples'])/3000;var=sum((q[k]-mu)**2 for q in r['samples'])/2999
            self.assertLess(abs(mu),.08);self.assertLess(abs(var-1),.1)
    def test_invalid_inputs(self):
        for kwargs in [{'draws':True},{'draws':1},{'warmup':-1},{'mass':[0,1]},{'step_size':float('nan')},{'steps':0},{'seed':-1},{'method':'nuts'},{'draws':10000,'steps':1000}]:
            with self.assertRaises(ValueError):sample(standard,[0.,0.],**kwargs)
        for q in [[0.], [True,0.],[float('inf'),0.]]:
            with self.assertRaises(ValueError):leapfrog(standard,q,[0.,0.])
