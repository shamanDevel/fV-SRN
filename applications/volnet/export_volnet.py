"""
Exports the trained networks to the renderer
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import torch
import h5py
import argparse
import io
from typing import Union
from collections import OrderedDict

from tests.volnet.network import InputParametrization, OutputParametrization, SceneNetwork
from diffdvr.utils import renderer_dtype_torch
import pyrenderer

class NetworkLoader:

    def __init__(self, file: h5py.File, device: torch.device):
        output_mode_str = file.attrs["outputmode"]
        self.has_direction = file.attrs["useDirection"] if "useDirection" in file.attrs else False
        self.num_fourier_features = file.attrs["fouriercount"] if "fouriercount" in file.attrs else 0
        activation_str = file.attrs["activation"]

        # convert enums
        if output_mode_str == "density":
            self.output_mode = pyrenderer.OutputMode.Density
        elif output_mode_str == "color":
            self.output_mode = pyrenderer.OutputMode.Color
        else:
            raise ValueError("Unsupported output mode " + output_mode_str)
        print("output mode:", self.output_mode)

        if activation_str == "ReLU":
            self.activation = pyrenderer.Activation.ReLU
        elif activation_str == "Sigmoid":
            self.activation = pyrenderer.Activation.Sigmoid
        elif activation_str == "Tanh":
            self.activation = pyrenderer.Activation.Tanh
        elif activation_str == "Softplus":
            self.activation = pyrenderer.Activation.Softplus
        else:
            raise ValueError("Unsupported activatíon " + activation_str)
        print("Activation function:", self.activation)

        # prepare pytorch network
        self.layers = file.attrs['layers']
        self.input_parametrization = InputParametrization(
            has_direction=(file.attrs['mode'] == 'screen' and file.attrs['useDirection']),
            num_fourier_features=file.attrs["fouriercount"] if "fouriercount" in file.attrs else 0,
            fourier_std=file.attrs["fourierstd"] if "fourierstd" in file.attrs else 1)
        self.output_parametrization = OutputParametrization(
            output_mode=file.attrs['outputmode'])
        self.hidden_network = SceneNetwork(
            input_channels=self.input_parametrization.num_output_channels(),
            output_channels=self.output_parametrization.num_input_channels(),
            layers=file.attrs['layers'],
            activation=file.attrs['activation'])
        self.full_network = torch.nn.Sequential(OrderedDict([
            ('input', self.input_parametrization),
            ('hidden', self.hidden_network),
            ('output', self.output_parametrization)
        ]))
        self.full_network.to(device=device, dtype=renderer_dtype_torch)
        self.full_network.eval()
        self.device = device

    def fill_weights(self, weights, epoch):
        weights_np = weights[epoch, :]
        try:
            weights_bytes = io.BytesIO(weights_np.tobytes())
            self.full_network.load_state_dict(
                torch.load(weights_bytes, map_location=self.device), strict=True)
        except:
            print("Unable to fill full network, now just fill hidden network (old version)")
            weights_bytes = io.BytesIO(weights_np.tobytes())
            self.hidden_network.load_state_dict(
                torch.load(weights_bytes, map_location=self.device), strict=True)

    def create_scene_network(self):
        # setup input + output parametrization
        net = pyrenderer.SceneNetwork()
        net.input.has_direction = self.has_direction
        #net.input.num_fourier_features = self.num_fourier_features
        if self.num_fourier_features>0:
            net.input.set_fourier_features(
                self.input_parametrization.get_fourier_feature_matrix().cpu().float().numpy())
            assert net.input.num_fourier_features == self.num_fourier_features
        net.output.output_mode = self.output_mode

        # convert to the renderer struct
        num_layers = len(self.layers.split(':'))
        for i in range(num_layers + 1):
            layer: torch.nn.Linear = getattr(self.hidden_network, 'linear%d' % i)
            weights = layer.weight.float()
            # print(weights.shape)
            bias = layer.bias.float()
            # print(bias.shape)
            activation = self.activation if i < num_layers else pyrenderer.Activation.NONE
            net.add_layer(weights, bias, activation)

        if not net.valid():
            raise ValueError("Network configuration is invalid")

        return net

def result_to_network(file: Union[str, h5py.File], epoch: int) -> pyrenderer.SceneNetwork:
    if isinstance(file, str):
        with h5py.File(file, "r") as f:
            return result_to_network(f, epoch)
    assert isinstance(file, h5py.File)

    loader = NetworkLoader(file, torch.device('cpu'))
    loader.fill_weights(file['weights'], epoch)
    return loader.create_scene_network()

def __main():
    parser = argparse.ArgumentParser(
        description='Training output to scene network',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input', type=str, help="Input .hdf5 file")
    parser.add_argument('-e', '--epoch', type=int, default=-1, help="The epoch to use")
    parser.add_argument('output', type=str, nargs='?', default=None, help="Output file for the scene network")
    opt = parser.parse_args()
    if opt.output is None:
        opt.output = os.path.splitext(opt.input)[0] + '.volnet'

    net = result_to_network(opt.input, opt.epoch)
    print("Converted, save to", opt.output)
    net.save(opt.output)

if __name__ == '__main__':
    __main()

