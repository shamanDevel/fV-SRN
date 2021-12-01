import unittest

import torch

import common.utils as utils
import pyrenderer

"""
Test the pytorch extension functions
"""

class MyTestCase(unittest.TestCase):
    def test_interp1d(self):
        B = 2
        C = 4
        N = 3
        M = 8

        device = torch.device('cuda')
        fp = torch.rand((B,C,N), dtype=torch.float64, device=device)
        x = torch.rand((B,M), dtype=torch.float64, device=device) * (N+2) - 1
        fp.requires_grad_(True)
        x.requires_grad_(True)

        print("fp:\n",fp.detach().cpu().numpy())
        print("x:\n",x.detach().cpu().numpy())

        torch.autograd.gradcheck(pyrenderer.interp1D, (fp, x))


if __name__ == '__main__':
    unittest.main()
