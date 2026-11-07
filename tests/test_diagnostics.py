import unittest
from hamiltonlab.diagnostics import moments,classic_rhat

class DiagnosticTests(unittest.TestCase):
    def test_moments(self):
        r=moments([[-1.,-2.],[1.,2.]]);self.assertEqual(r['mean'],[0.,0.]);self.assertEqual(r['sample_covariance'],[[2.,4.],[4.,8.]])
        self.assertEqual(r['x_sign_transitions'],1)
    def test_classic_formula(self):
        r=classic_rhat([[[-1.,-1.],[1.,1.]],[[1.,1.],[3.,3.]]])
        self.assertAlmostEqual(r['coordinates'][0]['value'],(1.5)**.5,14)
    def test_constant_undefined_not_success(self):
        r=classic_rhat([[[1.,1.]]*5,[[2.,2.]]*5])
        self.assertIsNone(r['coordinates'][0]['value']);self.assertEqual(r['coordinates'][0]['status'],'undefined_zero_within_variance')
    def test_no_clamp_and_chain_permutation(self):
        c=[[[-1.,-1.],[1.,1.]]]*2;r=classic_rhat(c)
        self.assertLess(r['coordinates'][0]['value'],1.);self.assertEqual(r,classic_rhat(c[::-1]))
    def test_invalid(self):
        for c in [[],[[[0.,0.]]]*2,[[[0.,0.],[1.,1.]],[[0.,0.]]]]:
            with self.assertRaises(ValueError):classic_rhat(c)
