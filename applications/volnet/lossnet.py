import torch
import torch.nn as nn
import argparse

import losses.lossbuilder

class LossFactory:

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        parser_group = parser.add_argument_group("Loss")
        parser_group.add_argument('-lm', '--lossmode', choices=["density", "rgbo"],
                                  type=str, default=None, help="""
                The channels where the loss functions are applied for world-space training.
                Either 'density' (loss on densities) or 'rgbo' (loss on rgb + opacity/absorption).
                With 'rgbo', the parameter 'absorption_weighting' specifies the scaling factor of the absorption/alpha.

                If the output of the network is 'density', but the loss acts on 'color',
                the outputs are first sent through the TF to obtain the colors.
                It is invalid to have a network predicting 'color', but the loss acting on 'color'.

                If not specified, delegates to 'outputmode', the output of the network.
                """)
        parser_group.add_argument('-l1', default=0, type=float,
                                  help="[world+screen] Weight of the L1 loss")
        parser_group.add_argument('-l2', default=0, type=float,
                                  help="[world+screen] Weight of the L2 loss")
        parser_group.add_argument('--dssim', default=0, type=float,
                                  help="[screen only] Weight of the dSSIM image loss")
        parser_group.add_argument('--lpips', default=0, type=float,
                                  help="[screen only] Weight of the LPIPS image loss")
        parser_group.add_argument('--absorption_weighting', type=float, default=0.1,
                                  help="[world only] Weighting for the absorption term")
        parser_group.add_argument('--multiply_alpha', action='store_true',
                                  help="[screen only] pre-multiply alpha into colors before applying losses")

    @staticmethod
    def createLosses(opt: dict, dtype, device):
        """
        Creates the losses from the dictionary obtained from the ArgumentParser.
        Returns a tuple (loss-screen, loss-world, world-mode) where
         - loss-screen is the configured LossNetScreen instance
         - loss-world is the configured LossNetWorld instance
         - world-mode is the loss mode for world-space training, either 'density' or 'rgbo'
        :param opt: the dictionary with the results from the ArgumentParser
        :return: the tuple (loss-screen, loss-world, world-mode)
        """
        mode = opt['lossmode']
        l1 = opt['l1']
        l2 = opt['l2']
        dssim = opt['dssim']
        lpips = opt['lpips']
        multiply_alpha = opt['multiply_alpha']
        absorption_weighting = opt['absorption_weighting']
        return \
            LossNetScreen(l1, l2, dssim, lpips, multiply_alpha, device), \
            LossNetWorld(mode, l1, l2, absorption_weighting, device), \
            mode

class LossNetScreen(nn.Module):
    def __init__(self, l1: float, l2: float,
                 dssim: float=0, lpips: float=0,
                 multiply_alpha: bool = False,
                 device: torch.device = None):
        """
        Constructs the loss network.
        The inputs are assumed to be of shape B*C*H*W*C
        where C=4 with the channels interpreted as red-green-blue-alpha.

        :param l1: weighting for the l1 loss
        :param l2: weighting for the l2 loss
        :param multiply_alpha: if True and mode=='rgba',
            the rgb-colors of the prediction and reference are multiplied
            by the alpha of the reference -> Penalizes differences in color
            only where visible
        """
        super().__init__()

        self._color_channels = 3 if multiply_alpha else 4
        self._l1 = nn.L1Loss()
        self._l2 = nn.MSELoss()
        if device is None:
            device = torch.device("cpu")
        self._dssim = losses.lossbuilder.LossBuilder(device).dssim_loss(self._color_channels)
        self._lpips = losses.lossbuilder.LossBuilder(device).lpips_loss(self._color_channels, 0.0, 1.0)

        assert l1>0 or l2>0 or dssim>0 or lpips>0, "at least one loss must be active"
        self._l1weight = l1
        self._l2weight = l2
        self._ssimweight = dssim
        self._lpipsweight = lpips
        self._multiply_alpha = multiply_alpha

    def loss_names(self):
        return ["l1", "l2", "dssim", "lpips", "total"]

    def forward(self, prediction, reference, return_individual_losses = False):
        assert len(reference.shape) == 4  # BCHW
        assert reference.shape[1] == 4
        assert reference.shape == prediction.shape
        if self._multiply_alpha:
            alpha = reference[:, 3:, :, :]
            prediction = torch.cat([
                    prediction[:, :3, :, :] * alpha,
                    prediction[:, 3:, :, :]
                ], dim=1)
            reference = torch.cat([
                reference[:, :3, :, :] * alpha,
                alpha
            ], dim=1)

        l1 = self._l1(prediction, reference)
        l2 = self._l2(prediction, reference)
        if self._ssimweight>0 or (prediction.requires_grad==False and reference.requires_grad==False):
            # test or training with weight
            ssim = self._dssim(prediction[:,:self._color_channels,:,:], reference[:,:self._color_channels,:,:])
        else:
            ssim = torch.zeros((1,), device=prediction.device, dtype=prediction.dtype)
        if self._lpipsweight > 0 or (prediction.requires_grad == False and reference.requires_grad == False):
            # test or training with weight
            lpips = self._lpips(prediction[:,:self._color_channels,:,:], reference[:,:self._color_channels,:,:])
        else:
            lpips = torch.zeros((1,), device=prediction.device, dtype=prediction.dtype)
        total = None
        if self._l1weight>0:
            total = (self._l1weight * l1) if total is None else (total + self._l1weight * l1)
        if self._l2weight>0:
            total = (self._l2weight * l2) if total is None else (total + self._l2weight * l2)
        if self._ssimweight>0:
            total = (self._ssimweight * ssim) if total is None else (total + self._ssimweight * ssim)
        if self._lpipsweight>0:
            total = (self._lpipsweight * lpips) if total is None else (total + self._lpipsweight * lpips)
        if return_individual_losses:
            return total, {
                'l1': l1.item(),
                'l2': l2.item(),
                'dssim': ssim.item(),
                'lpips': lpips.item(),
                'total': total.item()
            }
        else:
            return total


class LossNetWorld(nn.Module):
    def __init__(self, mode:str, l1: float, l2: float,
                 absorption_weight: float = 1,
                 device: torch.device = None):
        """
        Constructs the loss network.
        The inputs are assumed to be of shape B*C*H*W*C
        where C=4 if mode=='rgbo' with the channels interpreted as red-green-blue-opacity,
        or C=1 if mode=='density'

        :param l1: weighting for the l1 loss
        :param l2: weighting for the l2 loss
        :param absorption_weight: weights the opacity/absorption for mode 'rgbo'
        """
        super().__init__()
        assert mode in ['rgbo', 'density'], \
            f"mode must be 'rgbo' or 'density', but is {mode}"

        self._l1 = nn.L1Loss()
        self._l2 = nn.MSELoss()

        assert l1>0 or l2>0, "at least one loss must be active"
        self._l1weight = l1
        self._l2weight = l2
        self._absorption_weighting = absorption_weight
        self._mode = mode

    def mode(self):
        return self._mode

    def loss_names(self):
        if self._mode == 'density':
            return ["l1", "l2", "total"]
        else:
            return ["l1rgb", "l1alpha", "l2rgb", "l2alpha", "total"]

    def forward(self, prediction, reference,
                return_individual_losses = False):
        if self._mode == 'density':
            l1 = self._l1(prediction, reference)
            l2 = self._l2(prediction, reference)
            total = None
            if self._l1weight>0:
                total = (self._l1weight * l1) if total is None else (total + self._l1weight * l1)
            if self._l2weight>0:
                total = (self._l2weight * l2) if total is None else (total + self._l2weight * l2)
            if return_individual_losses:
                return total, {
                    'l1': l1.item(),
                    'l2': l2.item(),
                    'total': total.item()
                }
            else:
                return total
        else:
            assert prediction.shape[-1] == 4
            x_rgb = prediction[..., :3]
            x_alpha = prediction[..., 3:]
            y_rgb = reference[..., :3]
            y_alpha = reference[..., 3:]
            l1rgb = self._l1(x_rgb, y_rgb)
            l1alpha = self._l1(x_alpha, y_alpha)
            l2rgb = self._l2(x_rgb, y_rgb)
            l2alpha = self._l2(x_alpha, y_alpha)
            total = None
            if self._l1weight > 0:
                total = (self._l1weight * l1rgb) if total is None else (total + self._l1weight * l1rgb)
                total = total + (self._l1weight * self._absorption_weighting) * l1alpha
            if self._l2weight > 0:
                total = (self._l2weight * l2rgb) if total is None else (total + self._l2weight * l2rgb)
                total = total + (self._l2weight * self._absorption_weighting) * l2alpha
            if return_individual_losses:
                return total, {
                    'l1rgb': l1rgb.item(),
                    'l1alpha': l1alpha.item(),
                    'l2rgb': l2rgb.item(),
                    'l2alpha': l2alpha.item(),
                    'total': total.item()
                }
            else:
                return total
