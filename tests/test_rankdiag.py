import copy
import json
import math
import subprocess
import sys
import unittest
from statistics import NormalDist
from hamiltonlab.rankdiag import normal_scores,rank_split_rhat,scalar_rhat

class RankTests(unittest.TestCase):
    def test_tied_ranks_and_blom(self):
        r=normal_scores([3.,1.,1.,2.]);self.assertEqual(r['average_ranks'],[4.,1.5,1.5,3.])
        for rank,z in zip(r['average_ranks'],r['normal_scores']):self.assertAlmostEqual(NormalDist().cdf(z),(rank-.375)/4.25,14)
    def test_all_ties_and_symmetry(self):
        self.assertEqual(normal_scores([2.]*6)['normal_scores'],[0.]*6)
        z=normal_scores([0.,1.,2.,3.])['normal_scores'];self.assertAlmostEqual(z[0],-z[3],14);self.assertAlmostEqual(z[1],-z[2],14)
    def test_scalar_closed_reference(self):
        r=scalar_rhat([[-1.,1.],[1.,3.]])
        self.assertEqual(r['within_chain_variance'],2.);self.assertEqual(r['between_chain_variance'],4.);self.assertAlmostEqual(r['value'],math.sqrt(1.5),14)
    def test_split_order_and_fold(self):
        chains=[[[x,2*x] for x in [0.,1.,2.,3.]],[[x,2*x] for x in [4.,5.,6.,7.]]]
        r=rank_split_rhat(chains,True);a=r['coordinates'][0]['audit']
        self.assertEqual(a['raw_split_order'],[0.,1.,2.,3.,4.,5.,6.,7.]);self.assertEqual(a['folded_split_order'],[3.5,2.5,1.5,.5,.5,1.5,2.5,3.5]);self.assertEqual(r['dropped_draws'],0);self.assertEqual(r['split_chains'],4)
        self.assertEqual(a['folded_rank']['average_ranks'],[7.5,5.5,3.5,1.5,1.5,3.5,5.5,7.5])
    def test_invariance_and_no_mutation(self):
        c=[[[x,x+.2] for x in [0.,3.,1.,2.,4.,7.,5.,6.]],[[x,x+.2] for x in [.5,3.5,1.5,2.5,4.5,7.5,5.5,6.5]]];old=copy.deepcopy(c)
        r=rank_split_rhat(c);q=rank_split_rhat([[[2*z+3 for z in p] for p in chain] for chain in c[::-1]])
        self.assertEqual(c,old)
        for a,b in zip(r['coordinates'],q['coordinates']):self.assertAlmostEqual(a['max_rhat'],b['max_rhat'],13)
        # Bulk component invariant to a strictly monotone non-affine transform. Folding need not be.
        q=rank_split_rhat([[[math.exp(z/10) for z in p] for p in chain] for chain in c])
        self.assertAlmostEqual(r['coordinates'][0]['rank_normalized_split']['value'],q['coordinates'][0]['rank_normalized_split']['value'],13)
    def test_degenerate_components_remain_undefined(self):
        r=rank_split_rhat([[[0.,0.]]*4,[[0.,0.]]*4]);self.assertIsNone(r['coordinates'][0]['max_rhat']);self.assertEqual(r['coordinates'][0]['status'],'undefined_component')
        r=rank_split_rhat([[[x,x] for x in [-1.,1.,-1.,1.]]]*2)
        self.assertIsNotNone(r['coordinates'][0]['rank_normalized_split']['value']);self.assertIsNone(r['coordinates'][0]['folded_rank_normalized_split']['value']);self.assertIsNone(r['coordinates'][0]['max_rhat'])
    def test_scale_and_half_shift_counterexamples(self):
        from hamiltonlab.diagnostics import classic_rhat
        c=[[[z*s,z] for z in [-.5,.5,-1.5,1.5]*25] for s in [1.,1.,10.,10.]]
        raw=classic_rhat(c)['coordinates'][0]['value'];r=rank_split_rhat(c)['coordinates'][0]
        self.assertLess(raw,1.01);self.assertLess(r['rank_normalized_split']['value'],1.01);self.assertGreater(r['folded_rank_normalized_split']['value'],1.2)
        c=[[[z,z] for z in [-2.,-1.]*25+[1.,2.]*25]]*4
        self.assertLess(classic_rhat(c)['coordinates'][0]['value'],1.01);self.assertGreater(rank_split_rhat(c)['coordinates'][0]['max_rhat'],1.2)
    def test_invalid(self):
        for c in [[],[[[0.,0.]]]*2,[[[0.,0.]]*5]*2,[[[float('nan'),0.]]*4]*2,[[[0.,0.]]*4,[[0.,0.]]*6],[[[0.,0.]]*2000]*64]:
            with self.assertRaises(ValueError):rank_split_rhat(c)
        with self.assertRaises(ValueError):normal_scores([True,1.])
    def test_cli(self):
        r=subprocess.run([sys.executable,'-m','hamiltonlab','diagnose','examples/chains.json'],capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr);self.assertEqual(json.loads(r.stdout)['split_chains'],4)
