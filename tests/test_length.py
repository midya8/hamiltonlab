import json
import math
import random
import subprocess
import sys
import unittest
from hamiltonlab.core import sample
from hamiltonlab.targets import standard,mixture

class LengthTests(unittest.TestCase):
    def test_backward_compatible_default(self):
        a=sample(standard,[1.,0.],draws=30,seed=5);b=sample(standard,[1.,0.],draws=30,seed=5,steps_range=None)
        self.assertEqual(a,b);self.assertNotIn('steps_range',a);self.assertNotIn('chosen_steps',a['transitions'][0])
    def test_determinism_bounds_histogram_and_work(self):
        kwargs=dict(draws=100,warmup=10,seed=3,steps_range=[1,4])
        a=sample(standard,[1.,0.],**kwargs);b=sample(standard,[1.,0.],**kwargs);self.assertEqual(a,b)
        self.assertEqual(sum(a['length_counts'].values()),110);self.assertTrue(all(v>0 for v in a['length_counts'].values()))
        self.assertEqual(a['potential_gradient_callback_calls'],1+sum(t['chosen_steps']+1 for t in a['transitions']))
        self.assertTrue(all(1<=t['chosen_steps']<=4 for t in a['transitions']));self.assertIsNone(a['steps'])
    def test_exact_state_independent_rng_sequence(self):
        # A standard quadratic never aborts; manually consume a length, two Gaussian momenta and uniform.
        r=sample(standard,[0.,0.],draws=20,warmup=0,seed=42,steps_range=[2,7]);rng=random.Random(42);expected=[]
        for _ in range(20):
            expected.append(rng.randint(2,7));rng.gauss(0,1);rng.gauss(0,1);rng.random()
        self.assertEqual(expected,[t['chosen_steps'] for t in r['transitions']])
        q=sample(standard,[100.,-100.],draws=20,warmup=0,seed=42,steps_range=[2,7])
        self.assertEqual(expected,[t['chosen_steps'] for t in q['transitions']])
    def test_singleton_range_and_no_mutation(self):
        bounds=[3,3];r=sample(standard,[1.,0.],draws=5,warmup=0,steps_range=bounds)
        self.assertEqual(bounds,[3,3]);bounds[0]=1;self.assertEqual(r['steps_range'],[3,3]);self.assertEqual(r['length_counts'],{'3':5})
    def test_resonance_motion_and_rejection_repeats(self):
        r=sample(standard,[2.,-1.],draws=200,warmup=0,step_size=math.sqrt(2),seed=71,steps_range=[1,4])
        self.assertGreater(max(q[0] for q in r['samples'])-min(q[0] for q in r['samples']),1.)
        rejected=[t for t in r['transitions'] if not t['accepted']];self.assertGreater(len(rejected),0)
        self.assertTrue(all(t['before']==t['state'] for t in rejected))
    def test_randomization_does_not_solve_multimodality(self):
        r=sample(mixture,[-6.,0.],draws=200,warmup=0,step_size=.15,seed=71,steps_range=[1,10])
        self.assertTrue(all(q[0]<0 for q in r['samples']))
    def test_failed_trajectory_actual_count(self):
        def bounded(q):
            if q[0]!=0.:raise ValueError('outside')
            return standard(q)
        r=sample(bounded,[0.,0.],draws=3,warmup=0,step_size=1.,seed=2,steps_range=[1,4])
        self.assertEqual(r['invalid_trajectories'],3);self.assertEqual(r['potential_gradient_callback_calls'],7)
    def test_invalid(self):
        for bounds in [[],[0,3],[4,1],[1,1001],[True,4],[1,2,3],'1..4']:
            with self.assertRaises(ValueError):sample(standard,[0.,0.],draws=5,steps_range=bounds)
        with self.assertRaises(ValueError):sample(standard,[0.,0.],method='random_walk',steps_range=[1,2])
        with self.assertRaises(ValueError):sample(standard,[0.,0.],draws=10000,steps_range=[1,1000])
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','sample','examples/random_length.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);v=json.loads(r.stdout);self.assertEqual(v['steps_range'],[1,4]);self.assertEqual(len(v['transitions']),200)
