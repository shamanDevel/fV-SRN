"""
Helpers to evaluate the scene representation networks in world- and screen space.
Input: Network and the current batch from the dataloader
"""

from typing import Optional
import torch

from volnet.network import SceneRepresentationNetwork
from volnet.lossnet import LossNetScreen, LossNetWorld
from volnet.training_data import TrainingData
from volnet.raytracing import Raytracing

import common.utils as utils
import pyrenderer

class EvaluateScreen:
    def __init__(self, network:SceneRepresentationNetwork, evaluator:pyrenderer.IImageEvaluator,
                 loss: Optional[LossNetScreen],
                 image_width:int, image_height:int, train:bool,
                 disable_inversion_trick: bool,
                 dtype, device):

        self._network = network
        self._use_checkpointing = False if disable_inversion_trick else train
        self._loss = loss

        self._network_output_mode = network.output_mode().split(':')[0]  # trim options
        if self._network_output_mode == 'density' and train:
            raise ValueError( "For now, only rgbo-output is supported in screen-space for networks in training mode, no training through a TF yet")
        self._raytracing = Raytracing(evaluator, self._network_output_mode, 1.0, image_width, image_height, dtype, device)

    def __call__(self, dataloader_batch):
        """
        Evaluates the network.
        If loss is None, the returns total_loss and partial_losses is also None
        :param dataloader_batch: the tuple of pytorch-cuda tensors from the screen-space dataloader
        :param loss: the screen-space loss network
        :return: image (B,4,H,W), totol_loss, partial_losses
        """

        camera, target, tf_index, time_index, ensemble_index, stepsize = dataloader_batch
        if isinstance(stepsize, torch.Tensor):
            stepsize = stepsize[0].item()
        self._raytracing.set_stepsize(stepsize)
        if self._use_checkpointing:
            image = self._raytracing.checkpointed_trace(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])
        else:
            image = self._raytracing.full_trace_forward(self._network, camera, network_args=[tf_index, time_index, ensemble_index, 'screen'])

        if self._loss is None:
            total_loss = None
            partial_losses = None
        else:
            total_loss, partial_losses = self._loss(image, target, return_individual_losses=True)

        return image, total_loss, partial_losses

class EvaluateWorld:
    def __init__(self, network:SceneRepresentationNetwork, evaluator:pyrenderer.IImageEvaluator,
                 loss: Optional[LossNetWorld], dtype, device):
        self._network = network
        self._loss = loss

        self._network_output_mode = network.output_mode().split(':')[0] # trim options
        self._loss_input_mode = self._network_output_mode if loss is None else loss.mode()

        assert self._network_output_mode in ['density', 'rgbo']
        assert self._loss_input_mode in ['density', 'rgbo']

        if self._network_output_mode=='density' and self._loss_input_mode=='rgbo':
            raise NotImplementedError("Training through the TF is not supported yet")
        if self._network_output_mode=='rgbo' and self._loss_input_mode=='density':
            raise ValueError("The loss function expects densities, but the network already predictions derived colors")

        if network.use_direction():
            raise ValueError("The network requires directions, but this is not available in world-space evaluation")

    def __call__(self, dataloader_batch):
        """
        Evaluates the current batch
        :param dataloader_batch: the batch from the world-space data loader
        :return: values (B,C), total_loss, partial_losses
        """

        position, target, tf, time, ensemble = dataloader_batch
        # evaluate network
        predictions = self._network(position, tf, time, ensemble, 'world')
        # loss
        if self._loss is None:
            total_loss = None
            partial_losses = None
        else:
            total_loss, partial_losses = self._loss(predictions, target, return_individual_losses=True)

        return predictions, total_loss, partial_losses