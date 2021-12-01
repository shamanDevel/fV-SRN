import unittest
import torch
import torch.nn.functional as F
import numpy as np
import os

from volnet.raytracing import Raytracing
from volnet.network import SceneRepresentationNetwork
import common.utils as utils
import pyrenderer

class TestRaytracing(unittest.TestCase):

    def setUp(self) -> None:
        settings_file = os.path.abspath(
            os.path.join(os.path.split(__file__)[0], '..', 'neuraltextures', 'config-files', 'c60-v1-linear.json'))
        print("Load settings from", settings_file)
        self._image_evaluator = pyrenderer.load_from_json(settings_file)

        self._device = torch.device('cuda')
        self._dtype = torch.float32
        self._raytracing = Raytracing(self._image_evaluator, "rgbo",
                                      0.1, 16, 24, self._dtype, self._device)
        self._raytracing.set_stepsize(0.05)

    def test_utils(self):
        inBCHW = torch.rand((2, 4, 8, 12), dtype=self._dtype)
        xBHWC = utils.toHWC(inBCHW)
        outBCHW = utils.toCHW(xBHWC)
        np.testing.assert_allclose(inBCHW.cpu().numpy(), outBCHW.cpu().numpy())

    def test_blending_inverse(self):
        B = 16
        # inputs
        prev_color = torch.rand((B, 3), dtype=torch.float64)
        prev_alpha = torch.rand((B, 1), dtype=torch.float64)
        current_color = torch.rand((B, 4), dtype=torch.float64)
        mask = torch.rand((B, 1), dtype=torch.float64) > 0.3
        # forward
        next_color, next_alpha = self._raytracing._blend(
            prev_color, prev_alpha, current_color, mask)
        # inverse / backward
        grad_next_color = torch.randn((B, 3), dtype=torch.float64)
        grad_next_alpha = torch.randn((B, 1), dtype=torch.float64)
        new_prev_color, new_prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color = \
            self._raytracing._inverse_blend(next_color, next_alpha, current_color, mask,
                      grad_next_color, grad_next_alpha)

        self.assertIsNone(np.testing.assert_array_almost_equal(
            prev_color.numpy(), new_prev_color.numpy()))
        self.assertIsNone(np.testing.assert_array_almost_equal(
            prev_alpha.numpy(), new_prev_alpha.numpy()))

    def test_blending_gradient(self):
        class BlendingFunction(torch.autograd.Function):
            @staticmethod
            def forward(ctx, prev_color, prev_alpha, current_color, mask):
                next_color, next_alpha = self._raytracing._blend(
                    prev_color, prev_alpha, current_color, mask)
                ctx.save_for_backward(next_color, next_alpha, current_color, mask)
                return next_color, next_alpha

            @staticmethod
            def backward(ctx, grad_next_color, grad_next_alpha):
                next_color, next_alpha, current_color, mask = ctx.saved_tensors
                new_prev_color, new_prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color = \
                    self._raytracing._inverse_blend(next_color, next_alpha, current_color, mask,
                                              grad_next_color, grad_next_alpha)
                return grad_prev_color, grad_prev_alpha, grad_current_color, None

        B = 16
        # inputs
        prev_color = torch.rand((B, 3), dtype=torch.float64)
        prev_color.requires_grad_()
        prev_alpha = torch.rand((B, 1), dtype=torch.float64)
        prev_alpha.requires_grad_()
        current_color = torch.rand((B, 4), dtype=torch.float64)
        current_color.requires_grad_()
        mask = torch.rand((B, 1), dtype=torch.float64) >= 0.0

        fun = BlendingFunction.apply
        ret = torch.autograd.gradcheck(fun, (prev_color, prev_alpha, current_color, mask))
        self.assertTrue(ret, "gradient check failed")

    def test_checkpointed_trace(self):
        network = SceneRepresentationNetwork({
            'outputmode': 'rgbo',
            'use_direction': False,
            'time_features': 0,
            'ensemble_features': 0,
            'fourierstd': 1,
            'fouriercount': 0,
            'layers': '2:2',
            'activation': 'ReLU'
        }, self._dtype, self._device)
        network.to(self._device)
        network.train()

        optim = torch.optim.Adam(network.parameters(), lr=1)


        # full_trace_forward
        optim.zero_grad()
        output1 = self._raytracing.full_trace_forward(network)
        reference = torch.randn_like(output1).detach()
        loss = F.mse_loss(output1, reference)
        loss.backward()
        grads1 = [param.grad.detach().clone() for param in network.parameters()]
        output1 = output1.detach()

        # checkpointed_trace
        optim.zero_grad()
        output2 = self._raytracing.checkpointed_trace(network)
        loss = F.mse_loss(output2, reference)
        loss.backward()
        grads2 = [param.grad.detach().clone() for param in network.parameters()]
        output2 = output2.detach()

        # compare
        diff_output = torch.sum(output1-output2).item()
        print("Difference output:", diff_output)
        print("Max magnitude output:", torch.max(output1).item())
        np.testing.assert_allclose(output1.cpu().numpy(), output2.cpu().numpy(), atol=1e-4)
        for p1,p2 in zip(grads1, grads2):
            print(f"Parameter of shape {p1.shape}, difference: {torch.sum(p1-p2).item()}, magnitude: {torch.max(p1).item()}")
            np.testing.assert_allclose(p1.cpu().numpy(), p2.cpu().numpy(), atol=1e-4)

if __name__ == '__main__':
    torch.manual_seed(42)
    np.random.seed(42)
    unittest.TextTestRunner().run(TestRaytracing())