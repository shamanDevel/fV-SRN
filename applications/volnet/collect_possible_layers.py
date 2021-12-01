"""
Compute the possible layer configurations based
on the shared memory limitation
"""

import os
import numpy as np
import torch

import common.utils as utils
import pyrenderer
from volnet.network import SceneRepresentationNetwork

def compute_memory(channels, layers, fouriercount):
    layer_str = ':'.join([str(channels)] * (layers - 1))

    GRID_RESOLUTION = 32
    GRID_CHANNELS = 16

    opt = {
        'outputmode': 'density:direct',
        'layers': layer_str,
        'activation': 'ReLU',
        'fouriercount': fouriercount,
        'fourierstd': 1.0,
        'time_features': 0,
        'ensemble_features': 0,
        'volumetric_features_channels': GRID_CHANNELS,
        'volumetric_features_resolution': GRID_RESOLUTION,
        'volumetric_features_std': 0,
        'volumetric_features_time_dependent': False,
        'use_direction': False,
        'disable_direction_in_fourier_features': False,
        'fourier_position_direction_split': -1,
        'use_time_direct': False,
        'num_time_fourier': False,
        'meta_network': None,
        'meta_activation': None,
        'meta_pretrain': None
    }
    dtype = torch.float32
    device = torch.device('cuda')
    net = SceneRepresentationNetwork(opt, None, dtype, device)
    sn = net.export_to_pyrenderer(opt, pyrenderer.SceneNetwork.LatentGrid.Float)
    warps_shared = sn.compute_max_warps(False)
    warps_mixed = sn.compute_max_warps(True)
    return warps_shared, warps_mixed, layer_str

def collect_possible_layers():
    for c in [32,48,64,80,96,112,128]:
        l = 2
        while True:
            warps_shared, warps_mixed, layer_str = compute_memory(c, l, (c-4)//2)
            if warps_mixed>0:
                print(f"channels: {c}, layers: {l} -> {warps_mixed}/{warps_shared} warps, layer sting: {layer_str}")
            else:
                break
            l += 1

if __name__ == '__main__':
    print("Possible layers:")
    collect_possible_layers()