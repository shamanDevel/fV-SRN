"""
Optimizer settings for scene representation networks
"""

import numpy as np
import torch
import torch.nn.functional as F
import argparse
import json

class Optimizer:
    """
    Class to create and manage the optimizer
    """

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        parser_group = parser.add_argument_group("Optimization")
        parser_group.add_argument('-o', '--optimizer', default='Adam', type=str,
                                  help="The optimizer class, 'torch.optim.XXX'")
        parser_group.add_argument('-lr', default=0.01, type=float, help="The learning rate")
        parser_group.add_argument('-i', '--epochs', default=50, type=int,
                                  help="The number of iterations in the training")
        parser_group.add_argument('--lr_gamma', type=float, default=0.5,
                                  help='The learning rate decays every lrStep-epochs by this factor')
        parser_group.add_argument('--lr_step', type=int, default=500,
                                  help='The learning rate decays every lrStep-epochs (this parameter) by lrGamma factor')
        parser_group.add_argument('--optim_params', default="{}", type=str,
                                  help="Additional optimizer parameters parsed as json")

    def __init__(self, opt: dict, parameters, dtype, device):
        self._opt = opt
        self._optimizer_class = getattr(torch.optim, opt['optimizer'])
        self._optimizer_parameters = json.loads(opt['optim_params'])
        self._optimizer_parameters['lr'] = opt['lr']
        self._optimizer = self._optimizer_class(parameters, **self._optimizer_parameters)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, opt['lr_step'], opt['lr_gamma'])
        self._num_epochs = opt['epochs']

    def reset(self, parameters):
        """
        Resets the optimizer and LR-scheduler
        """
        self._optimizer = self._optimizer_class(parameters, **self._optimizer_parameters)
        self._scheduler = torch.optim.lr_scheduler.StepLR(self._optimizer, self._opt['lr_step'], self._opt['lr_gamma'])

    def num_epochs(self):
        return self._num_epochs

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self, closure):
        self._optimizer.step(closure)

    def post_epoch(self):
        self._scheduler.step()

    def get_lr(self):
        return self._scheduler.get_last_lr()