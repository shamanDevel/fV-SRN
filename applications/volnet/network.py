"""
Volume encoding network
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import argparse
import math
from itertools import product
import tqdm

import common.utils as utils
import pyrenderer
from .input_data import TrainingInputData

class InputParametrization(nn.Module):
    # in earlier checkpoints, the factor 2pi was applied to the fourier matrix during 'forward'
    # In newer version, this is done directly in the constructor (more efficient).
    # For old checkpoints, this variable is set during loading to false
    PREMULTIPLY_2_PI = True

    def __init__(self, *,
                 has_direction = False,
                 num_fourier_features: int = 0,
                 fourier_std: float = 1,
                 disable_direction_in_fourier: bool = True,
                 fourier_position_direction_split: int = -1,
                 use_time_direct: bool = False,
                 num_time_fourier: int = 0):
        super().__init__()
        self._has_direction = has_direction
        self._num_fourier_features = num_fourier_features
        self._disable_direction_in_fourier = disable_direction_in_fourier if (disable_direction_in_fourier is not None) else True
        self._use_time_direct = use_time_direct or False
        self._num_time_fourier = num_time_fourier or 0
        self._fourier_position_direction_split = fourier_position_direction_split or -1
        self._premultiply2pi = InputParametrization.PREMULTIPLY_2_PI
        if num_fourier_features>0:
            out = 6 if (has_direction and not disable_direction_in_fourier) else 3

            if self._num_time_fourier>0:
                num_position_fourier = num_fourier_features - self._num_time_fourier
            else:
                num_position_fourier = num_fourier_features

            if fourier_std>0:
                # random gaussian
                B = torch.normal(0, fourier_std, (num_position_fourier, out))
                if self._premultiply2pi:
                    B = B * (2*np.pi)
            else:
                # scaled block-identity, based on NeRF
                assert self._fourier_position_direction_split < 0, "fourier-split not compatible with NeRF-position-matrix"
                num_blocks = int(np.ceil(num_position_fourier/out))
                Bx = []
                for i in range(num_blocks):
                    Bx.append(((2**i)) * torch.eye(out, out))
                B = torch.cat(Bx, dim=0)[:num_position_fourier,:]
                if self._premultiply2pi:
                    B = B * (2*np.pi)
            if self._fourier_position_direction_split>=0:
                assert has_direction and not disable_direction_in_fourier
                assert self._fourier_position_direction_split<num_position_fourier
                # set directional component for [:fourier_position_direction_split] to zero
                B[:self._fourier_position_direction_split,3:].zero_()
                # set positional component for [fourier_position_direction_split:] to zero
                B[self._fourier_position_direction_split:, :3].zero_()
            self.register_buffer('B', B)

            if self._num_time_fourier > 0:
                if fourier_std > 0:
                    B_time = torch.normal(0, fourier_std, (self._num_time_fourier, 1))
                else:
                    B_time = torch.tensor([2*np.pi*(2**i) for i in range(self._num_time_fourier)], dtype=torch.float32)
                    B_time = B_time.unsqueeze(1)
                self.register_buffer('B_time', B_time)

        assert has_direction or not disable_direction_in_fourier #disable_direction_in_fourier implies has_direction

    def has_direction(self):
        return self._has_direction

    def has_position(self):
        return True

    def is_premultiplied(self):
        return self._premultiply2pi

    def has_time(self):
        return self._use_time_direct or self._num_time_fourier>0

    def num_input_channels(self):
        """
        Returns the number of input channels:
        3 for position (x,y,z)
        3 for direction if enabled (dx, dy, dz)
        :return: the number of input channels
        """
        return 3 + (3 if self._has_direction else 0) + (1 if self.has_time() else 0)

    def _num_direct_output_channels(self):
        """
        Returns the number of input channels that are directly passed on to the output
        """
        return 3 + (3 if self._has_direction else 0) + (1 if self._use_time_direct else 0)

    def num_output_channels(self):
        """
        :return: the number of output channels
        """
        out = 3 + (3 if self._has_direction else 0) + (1 if self._use_time_direct else 0)
        return out + 2*self._num_fourier_features

    def get_fourier_feature_matrix(self):
        """
        :return: the fourier feature matrix of shape B*3 or None if no fourier features are here
        """
        return self.B if self._num_fourier_features>0 else None

    def forward(self, x):
        """
        Input parametrization from (B, Cin) to (B, Cout)
        where Cin=self.num_input_channels(), Cout=self.num_output_channels().
        Any additional channels are simply added to the end,
        use this for latent vectors for timestep, ensemble, TF.
        """
        assert len(x.shape)==2, \
            "input is not of shape (B,Cin), but " + str(x.shape)

        extra_channels = x.shape[1] - self.num_input_channels()
        assert extra_channels>=0, f"At least {self.num_input_channels()} channels expected, but got {x.shape[1]}"

        if self._num_fourier_features > 0:
            if self._has_direction and self._disable_direction_in_fourier:
                x_base = x[:,:self._num_direct_output_channels()]
                x_fourier = x[:,:3]
                x_extra = x[:,self.num_input_channels():]
            elif self._has_direction and not self._disable_direction_in_fourier:
                x_base = x[:, :self._num_direct_output_channels()]
                x_fourier = x[:, :6]
                x_extra = x[:, self.num_input_channels():]
            else: # not self._has_direction:
                x_base = x[:, :self._num_direct_output_channels()]
                x_fourier = x[:, :3]
                x_extra = x[:, self.num_input_channels():]
            f = torch.matmul(self.B, x_fourier.t()).t()
            if self._premultiply2pi:
                f2 = f
            else:
                f2 = 2 * np.pi * f

            x_parts = [
                x_base,
                torch.cos(f2),
                torch.sin(f2)]

            if self._num_time_fourier>0:
                x_time = x[:,3:4]
                ftime = torch.matmul(self.B_time, x_time.t()).t() # B_time has the factor 2*pi backed in already
                x_parts.append(torch.cos(ftime))
                x_parts.append(torch.sin(ftime))

            x_parts.append(x_extra)
            x = torch.cat(x_parts, dim=1)

        return x

class OutputParametrization(nn.Module):
    DENSITY = "density"
    RGBO = "rgbo"
    DENSITY_DIRECT = "density:direct"
    RGBO_DIRECT = "rgbo:direct"
    RGBO_EXP = "rgbo:exp"

    def __init__(self, output_mode:str):
        """
        Output parametrization.
        :param output_mode:
         - density: estimates a single density in [0,1]
         - color: estimates color in [0,1] + absorption in [0,infty]
        """
        super().__init__()
        assert output_mode in [
                OutputParametrization.DENSITY,
                OutputParametrization.DENSITY_DIRECT,
                OutputParametrization.RGBO,
                OutputParametrization.RGBO_DIRECT,
                OutputParametrization.RGBO_EXP], \
            "output_mode must be 'density' or 'rgb', but is %s"%output_mode
        self._output_mode = output_mode

    def num_input_channels(self):
        if self._output_mode in [OutputParametrization.DENSITY, OutputParametrization.DENSITY_DIRECT]:
            return 1
        else:
            return 4

    def num_output_channels(self):
        return self.num_input_channels()

    def forward(self, x, mode='screen'):
        """
        Output parametrization from (B, Cin) to (B, Cout)
        where Cin=self.num_input_channels(), Cout=self.num_output_channels()
        """
        assert len(x.shape) == 2, \
            "input is not of shape (B,Cin), but" + str(x.shape)
        assert x.shape[1] == self.num_input_channels(), \
            "invalid number of input channels, expected %d but got %d" % (self.num_input_channels(), x.shape[1])
        assert mode in ["world", "screen"]

        if self._output_mode == OutputParametrization.DENSITY:
            return torch.sigmoid(x)
        elif self._output_mode == OutputParametrization.DENSITY_DIRECT:
            if mode=='screen':
                return torch.clamp(x, min=0, max=1)
            else:
                return x
        else:
            rgb = x[...,:3]
            absorption = x[...,3:]
            if self._output_mode == OutputParametrization.RGBO:
                rgb = torch.sigmoid(rgb)
                absorption = F.softplus(absorption)
            elif self._output_mode == OutputParametrization.RGBO_DIRECT:
                if mode=='screen':
                    rgb = torch.clamp(rgb, min=0, max=1)
                    absorption = torch.clamp(absorption, min=0)
                else:
                    pass # do nothing
            elif self._output_mode == OutputParametrization.RGBO_EXP:
                rgb = torch.sigmoid(rgb)
                absorption = torch.exp(absorption)
            return torch.cat((rgb, absorption), dim=-1)

class CustomActivations:
    class Sine(nn.Module):
        def __init__(self, w0=1):
            super().__init__()
            self.w0 = float(w0)

        def forward(self, x):
            return torch.sin(self.w0 * x)

    class Snake(nn.Module):
        def __init__(self, f=1):
            super().__init__()
            self.f = float(f)
        def forward(self, x):
            return x + (1./self.f) * (torch.sin(self.f * x)**2)

    class SnakeAlt(nn.Module):
        def __init__(self, f=1):
            super().__init__()
            self.f = float(f)
        def forward(self, x):
            t = x + 1 - torch.cos(2*self.f * x)
            return t/(2.*self.f)

    class ModulatedSine(nn.Module):
        def __init__(self, input_channels, output_channels, latent_size, is_first, w0=1):
            super().__init__()
            self._w0 = w0
            self._lat = latent_size
            self._isfirst = is_first
            self._relu = torch.nn.ReLU()
            if is_first:
                self._lin1 = torch.nn.Linear(
                    input_channels - latent_size, output_channels, True) # synthesizer
                self._lin2 = torch.nn.Linear(
                    latent_size, output_channels, True) # modulator
                self._isize = input_channels - latent_size
            else:
                self._lin1 = torch.nn.Linear(
                    input_channels, output_channels, True)  # synthesizer
                self._lin2 = torch.nn.Linear(
                    input_channels+latent_size, output_channels, True)  # modulator
                self._isize = input_channels
        def forward(self, x):
            if self._isfirst:
                i = x[:,:self._isize]
                z = x[:,self._isize:]
                hz = z
            else:
                i = x[:, :self._isize]
                hz = x[:, self._isize:]
                z = x[:,-self._lat:]

            new_h = self._relu(self._lin2(hz))
            new_i = new_h * torch.sin(self._lin1(i))

            res = torch.cat((new_i, new_h, z), dim=1)
            return res

    class Select(nn.Module):
        def __init__(self, _from, _to):
            super().__init__()
            self._from = _from
            self._to = _to
        def forward(self, x):
            return x[:,self._from:self._to]

class ResidualSineLayer(nn.Module):
    """
    From Lu & Berger 2021, Compressive Neural Representations of Volumetric Scalar Fields
    https://github.com/matthewberger/neurcomp/blob/main/siren.py
    """
    def __init__(self, features:int, bias=True, ave_first=False, ave_second=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0

        self.features = features
        self.linear_1 = nn.Linear(features, features, bias=bias)
        self.linear_2 = nn.Linear(features, features, bias=bias)

        self.weight_1 = .5 if ave_first else 1
        self.weight_2 = .5 if ave_second else 1

        self.init_weights()
    #

    def init_weights(self):
        with torch.no_grad():
            self.linear_1.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)
            self.linear_2.weight.uniform_(-np.sqrt(6 / self.features) / self.omega_0,
                                           np.sqrt(6 / self.features) / self.omega_0)
        #
    #

    def forward(self, input):
        sine_1 = torch.sin(self.omega_0 * self.linear_1(self.weight_1*input))
        sine_2 = torch.sin(self.omega_0 * self.linear_2(sine_1))
        return self.weight_2*(input+sine_2)
    #

class InnerNetwork(nn.Sequential):

    def __init__(self,
                 input_channels:int,
                 output_channels:int,
                 layers:str,
                 activation:str,
                 latent_size:int):
        """
        :param input_channels: InputParametrization.num_output_channels()
        :param output_channels: OutputParametrization.num_input_channels()
        :param layers: colon-separated list of hidden layer sizes
        :param activation: activation function, torch.nn.**
        :param latent_size: the size of the latent vector (for modulated sine)
        """

        layer_sizes = list(map(int, layers.split(':')))
        activationX = activation.split(':')
        activation_params = activationX[1:]
        layers = []
        last_channels = input_channels
        if activationX[0] == "ModulatedSine":
            # special handling for modulated sine
            for i, s in enumerate(layer_sizes):
                s = s // 2  # because modulated size doubles it for synthesizer+modulator
                # this way, the methods are comparable
                layers.append(('linear%d' % i, CustomActivations.ModulatedSine(
                    last_channels, s, latent_size, i == 0)))
                last_channels = s
            layers.append(('select_synthesizer', CustomActivations.Select(0, last_channels)))
            last_layer = nn.Linear(last_channels, output_channels)
        elif activationX[0] == "ResidualSine":
            # special handling for residual blocks
            if len(set(layer_sizes))!=1:
                raise ValueError("for ResidualSine, all layers must have the same size")
            hiddenSize = layer_sizes[0]
            # copied and modified from https://github.com/matthewberger/neurcomp/blob/main/siren.py
            for i, s in enumerate(layer_sizes):
                layer_in = last_channels
                layer_out = s
                if i==0:
                    _l = nn.Linear(layer_in, s)
                    with torch.no_grad():
                        _l.weight.uniform_(-1 / layer_in, 1 / layer_in)
                    layers.append(('linear%d'%i, _l))
                    layers.append(('Sine%d'%i, CustomActivations.Sine(w0=30)))
                else:
                    layers.append(('ResidualSine%d'%i, ResidualSineLayer(
                        s, bias=True, ave_first=i>1, ave_second=i==(len(layer_sizes)-2))
                    ))
                last_channels = layer_out
            last_layer = nn.Linear(last_channels, output_channels)
            with torch.no_grad():
                last_layer.weight.uniform_(-np.sqrt(6 / (last_channels)) / 30.0, np.sqrt(6 / (last_channels)) / 30.0)
        else:
            # normal activations
            activ_class = getattr(torch.nn, activationX[0], None)
            if activ_class is None:
                activ_class = getattr(CustomActivations, activationX[0])
            for i,s in enumerate(layer_sizes):
                layers.append(('linear%d'%i, nn.Linear(last_channels, s)))
                layers.append(('%s%d'%(activation.lower(), i), activ_class(*activation_params)))
                last_channels = s
            last_layer = nn.Linear(last_channels, output_channels)
        if output_channels==4: #rgba
            last_layer.bias.data = torch.abs(last_layer.bias.data) + 1.0 # positive output to see something
        #else:
        #    last_layer.weight.data = 100 * last_layer.weight.data
        layers.append(('linear%d' % len(layer_sizes), last_layer))


        super().__init__(collections.OrderedDict(layers))

        self._input_channels = input_channels
        self._output_channels = output_channels

    def num_input_channels(self):
        return self._input_channels

    def num_output_channels(self):
        return self._output_channels


class InnerNetworkMeta(nn.Module):
    """Inner network with meta-network for the weights"""

    def __init__(self,
                 input_channels:int,
                 output_channels:int,
                 layers_main:str,
                 activation_main:str,
                 layers_meta:str,
                 activation_meta:str,
                 latent_size:int,
                 enable_pretraining:bool):
        super().__init__()
        assert activation_main != "ModulatedSine", "ModulatedSine not supported with meta network"
        assert activation_meta != "ModulatedSine", "ModulatedSine not supported with meta network"

        self._input_channels = input_channels
        self._output_channels = output_channels
        self._layers_main = layers_main
        self._activation_main = activation_main
        self._activation_meta = activation_meta
        self._layers_meta = layers_meta
        self._latent_size = latent_size
        self._enable_pretraining = enable_pretraining

        # main scene representation network
        layers_main_sizes = list(map(int, layers_main.split(':')))
        activ_main_class = getattr(torch.nn, activation_main, None)
        if activ_main_class is None:
            activ_main_class = getattr(CustomActivations, activation_main)
        layers_descr = [] # tuples of (start_idx, end_idx_weights, end_idx_bias, input_shape, output_shape)
        activations = []
        num_parameters = 0
        last_channels = input_channels
        max_size = 0
        def add_layer(inc, outc, activ):
            nonlocal num_parameters, layers_descr, max_size
            layers_descr.append((
                num_parameters,
                num_parameters + inc * outc,
                num_parameters + (inc+1) * outc,
                inc,
                outc
            ))
            max_size = max(max_size, inc, outc)
            num_parameters += (inc+1) * outc
            activations.append(activ_main_class() if activ else None)
        for i, s in enumerate(layers_main_sizes):
            add_layer(last_channels, s, True)
            last_channels = s
        add_layer(last_channels, output_channels, False)
        print("The meta network has to predict", num_parameters, "parameters")
        self._layers_descr = layers_descr
        self._activations = activations
        self._num_parameters = num_parameters

        # meta network
        layers_meta_sizes = list(map(int, layers_meta.split(':')))
        activ_meta_class = getattr(torch.nn, activation_meta, None)
        if activ_meta_class is None:
            activ_meta_class = getattr(CustomActivations, activation_meta)
        layers_meta = []
        last_channels = latent_size
        # normal activations
        for i, s in enumerate(layers_meta_sizes):
            layers_meta.append(('linear%d' % i, nn.Linear(last_channels, s)))
            layers_meta.append(('relu%d' % (i), activ_meta_class()))
            last_channels = s
        last_layer = nn.Linear(last_channels, num_parameters)
        # because the last layer predicts the parameters for the SRN,
        # these values are used in the network multiplications and quickly explode in magnitude
        # scale them down!
        bound = 1 / (max_size * len(layers_descr))
        with torch.no_grad():
            last_layer.weight *= bound
            last_layer.bias *= 1 / (len(layers_descr))
        layers_meta.append(('linear%d' % len(layers_meta_sizes), last_layer))
        self._meta_network = nn.Sequential(collections.OrderedDict(layers_meta))

        if self._enable_pretraining:
            self.pretrain_parameters = torch.nn.parameter.Parameter(
                torch.Tensor(1, num_parameters))
            torch.nn.init.uniform_(self.pretrain_parameters, -bound, bound)


    def forward(self, latent_variables, points):
        """
        Evaluates the network.
        First, the weights for the actual scene network are computed
        from the latent variables.
        These weights are then used to evaluate the points
        :param latent_variables: the latent variables of shape (1,M) where M=latent_size
        :param points: the points to evaluate of shape (B,C) with B=batch, C=channels (3 or 6 typically)
        :return: the output of shape (B,output_channels)
        """
        assert latent_variables.shape == (1, self._latent_size)
        assert len(points.shape) == 2
        assert points.shape[1] == self._input_channels

        # meta network
        if self._enable_pretraining:
            params = self.pretrain_parameters
        else:
            params = self._meta_network(latent_variables)
        # main network
        x = points
        for layer_desc, activ in zip(self._layers_descr, self._activations):
            start_idx, end_idx_weights, end_idx_bias, input_shape, output_shape = layer_desc
            weight = params[0,start_idx:end_idx_weights].reshape(input_shape, output_shape)
            bias = params[0,end_idx_weights:end_idx_bias].reshape(output_shape)
            x = torch.addmm(bias, x, weight)
            if activ is not None:
                x = activ(x)
        return x

    def finish_pretraining(self, latent_variables: torch.Tensor, epochs: int):
        """
        Finishes the pretraining: The first phase where the meta-network is skipped is completed.
        Now train the meta-network to match the dummy parameters.
        :param input: the input latent variables (B,C)
        :param epochs: the number of epochs for the pretraining
        """
        assert self._enable_pretraining

        targets = self.pretrain_parameters.detach()
        latent_variables = latent_variables.detach() # (1, num_parameters)
        optimizer = torch.optim.Adam(self._meta_network.parameters(), lr=0.001)
        print("Train meta-network to match the pre-trained parameters")
        self._meta_network.train()
        with tqdm.tqdm(epochs) as iteration_bar:
            for epoch in range(epochs):
                optimizer.zero_grad()
                prediction = self._meta_network(latent_variables)
                loss = torch.nn.functional.mse_loss(prediction, targets)
                loss.backward()
                loss_value = loss.item()
                optimizer.step()
                iteration_bar.update(1)
                iteration_bar.set_description("Loss: %7.5f" % loss_value)
        optimizer.zero_grad(True)

        self._enable_pretraining = False
        print("Pretraining finished, resume with regular training")


class SceneRepresentationNetwork(nn.Module):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        parser_group = parser.add_argument_group("Network")
        parser_group.add_argument('-om', '--outputmode',
                                  choices=["density", "density:direct", "rgbo", "rgbo:direct", "rgbo:exp"],
                                  type=str, default="density", help="""
                        The possible outputs of the network:
                        - density: a scalar density is produced that is then mapped to color via the TF.
                          This allows to use multiple TFs, see option 'randomizeTF'.
                        - density:direct: noop for world-space, clamp to [0,1] for screen-space
                        - rgbo: the network directly estimates red,green,blue,opacity/absorption. 
                          The TF is fixed during training and inference.                      
                        - rgbo:direct: noop for world-space, clamp to [0,1] for color, [0,infty] for absorption
                          for screen-space
                        """)
        parser_group.add_argument('-l', '--layers', default='32:32:32', type=str,
                                  help="The size of the hidden layers, separated by colons ':'")
        parser_group.add_argument('-a', '--activation', default="ReLU", type=str, help="""
                        The activation function for the hidden layers.
                        This is the class name for activations in torch.nn.** .
                        The activation for the last layer is fixed by the output mode.
                        To pass extra arguments, separate them by colons, e.g. 'Snake:2'""")
        parser_group.add_argument('-fn', '--fouriercount', default=0, type=int,
                                  help="Number of fourier features")
        parser_group.add_argument('-fs', '--fourierstd', default=1, type=float, help="""
            Standard Deviation of the fourier features, a positive value.
            If a negative value, the special NeRF-compatibility mode is used where the fourier features
            are block-identity matrices scaled by power of twos.
            """)
        parser_group.add_argument('--time_features', default=0, type=int,
                                  help="Feature size for timestep encoding")
        parser_group.add_argument('--ensemble_features', default=0, type=int,
                                  help="Feature size for ensemble encoding")
        parser_group.add_argument('--volumetric_features_channels', default=0, type=int,
                                  help="For volumetric latent spaces, specify the channels per voxel here")
        parser_group.add_argument('--volumetric_features_resolution', default=0, type=int,
                                  help="For volumetric latent spaces, specify the grid resolution here")
        parser_group.add_argument('--volumetric_features_std', default=0.01, type=float,
                                  help="Standard deviation for sampling the initial volumetric features")
        parser_group.add_argument('--volumetric_features_time_dependent', action='store_true', help="""
            If specified, the volumetric feature grid is time+ensemble-dependent.
            The split between time features and ensemble features is controlled by
            '--time_features' and '--ensemble_features', they must sum up to 
            '--volumetric_features_channels'.
            This results in two 5D grids:
             - time grid of shape (time_features, num_timesteps, volumetric_features_resolution^3)
             - ensemble grid of shape (ensemble_features, num_timesteps, volumetric_features_resolution^3)                 
            """)
        parser_group.add_argument('--use_direction', action='store_true')
        parser_group.add_argument('--disable_direction_in_fourier_features', action='store_true')
        parser_group.add_argument('--fourier_position_direction_split', default=-1, type=int, help="""
            If specified with a value in {0,1,...,fouriercount-1},
            the fourier matrix is split between positional and directional part.
            The first 'fourier_position_direction_split' fourier features only act
            on the position, the other fourier features only act on the direction. 
            """)
        parser_group.add_argument('--use_time_direct', action='store_true', help="""
            Uses time as a direct (scalar) input to the network""")
        parser_group.add_argument('--num_time_fourier', type=int, default=0, help="""
            Allocates that many inputs from 'fouriercount' to time encoding instead of position.""")
        parser_group.add_argument('--meta_network', default=None, type=str, help="""
            Alternative way how TF/Ensemble/Time is encoded:
            The default, if this parameter is not specified, is to send the latent vectors
            for the TF/Ensemble/Time as additional input to the network.
            With this parameter, a second meta-network instead is first called that
            predicts the weights of the actual scene representation network from the latent vectors.
            This parameter specifies the hidden layers of that meta network, e.g. '64:64:64'.
            """)
        parser_group.add_argument('--meta_activation', default="ReLU", type=str, help="""
                        The activation function for the hidden layers in the meta network.
                        This is the class name for activations in torch.nn.** .
                        The activation for the last layer is fixed by the output mode.""")
        parser_group.add_argument('--meta_pretrain', default=None, type=str, help="""
            To improve stability, the meta-network can use a pre-training method.
            To enable this, specify this parameter with two integers "e1:e2".
            First, only the inner network is trained for e1 epochs with a set of parameters
            that is independent of the ensemble, time or TF. This allows the network
            to find a first coarse match.
            Then, the meta-network is trained to predict that set of parameters for
            all latent variables for e2 epochs.
            Only after that is the full pipeline trained end-to-end.
            """)

    def __init__(self, opt: dict, input: TrainingInputData, dtype, device):
        """
        Initializes the scene reconstruction network with the dictionary obtained from
        the ArgumentParser
        :param opt: the dictionary with the results from the ArgumentParser
        """
        super().__init__()
        self._dtype = dtype
        self._device = device

        self._outputmode = opt['outputmode']

        self._use_direction = opt['use_direction']
        self._time_features = opt['time_features']
        self._ensemble_features = opt['ensemble_features']
        self._meta_network_config = opt['meta_network']
        self._has_meta_network = self._meta_network_config is not None
        self._has_meta_pretraining = opt['meta_pretrain'] is not None
        self._disable_direction_in_fourier_features = opt['disable_direction_in_fourier_features']

        self._volumetric_features_channels = opt['volumetric_features_channels'] or 0
        self._volumetric_features_resolution = opt['volumetric_features_resolution'] or 0
        self._volumetric_features_time_dependent = opt['volumetric_features_time_dependent'] or False
        self._has_volumetric_features = self._volumetric_features_channels>0 and self._volumetric_features_resolution>0
        if not self._has_volumetric_features:
            self._volumetric_features_channels = 0 # so that the feature computation below is accurate
        if self._volumetric_features_time_dependent:
            assert self._has_volumetric_features, "A time-dependent volumetric feature grid is requested, but the resolution or channel count is zero"
            assert self._volumetric_features_channels==self._time_features+self._ensemble_features, \
                "A time-dependent volumetric feature grid is requested, but volumetric_features_channels!=time_features+ensemble_features"

        self._input_parametrization = InputParametrization(
            has_direction=self._use_direction,
            fourier_std=opt['fourierstd'], num_fourier_features=opt['fouriercount'],
            disable_direction_in_fourier=self._disable_direction_in_fourier_features,
            fourier_position_direction_split=opt['fourier_position_direction_split'],
            use_time_direct=opt['use_time_direct'], num_time_fourier=opt['num_time_fourier'])
        network_input_channels = self._input_parametrization.num_output_channels()
        self._output_parametrization = OutputParametrization(self._outputmode)

        self._base_input_channels = self._input_parametrization.num_input_channels()
        self._total_latent_size = \
            self._time_features + self._ensemble_features + \
            (self._volumetric_features_channels if not self._volumetric_features_time_dependent else 0)
        self._total_input_channels = self._base_input_channels + self._total_latent_size
        self._output_channels = self._output_parametrization.num_output_channels()

        if self._has_meta_network:
            self._hidden_layers = InnerNetworkMeta(
                network_input_channels,
                self._output_channels,
                opt['layers'], opt['activation'],
                opt['meta_network'], opt['meta_activation'],
                self._total_latent_size, self._has_meta_pretraining)
        else:
            self._hidden_layers = InnerNetwork(
                network_input_channels + self._total_latent_size,
                self._output_channels,
                opt['layers'], opt['activation'],
                self._total_latent_size)

        if self._has_meta_pretraining:
            s = opt['meta_pretrain'].split(':')
            assert len(s)==2
            self._meta_pretrain_epoch1 = int(s[0])
            self._meta_pretrain_epoch2 = int(s[1])
            self._meta_pretrain_current_epoch = 0
            self._meta_precompute_latent_variables(input)


        latent_space_memory = 0
        if self._has_volumetric_features:
            std = opt['volumetric_features_std']
            self._grid_std = std
            if self._volumetric_features_time_dependent:
                if self._time_features > 0:
                    p = torch.randn(
                        (input.num_timekeyframes(), self._time_features,
                         self._volumetric_features_resolution,
                         self._volumetric_features_resolution,
                         self._volumetric_features_resolution),
                    ) * std
                    self.register_parameter(
                        '_volumetric_latent_space_time',
                        torch.nn.parameter.Parameter(p))
                    latent_space_memory += p.numel()
                if self._ensemble_features > 0:
                    p = torch.randn(
                        (input.num_ensembles(), self._ensemble_features,
                         self._volumetric_features_resolution,
                         self._volumetric_features_resolution,
                         self._volumetric_features_resolution),
                    ) * std
                    self.register_parameter(
                        '_volumetric_latent_space_ensemble',
                        torch.nn.parameter.Parameter(p))
                    latent_space_memory += p.numel()
            else:
                p = torch.randn(
                    (1, self._volumetric_features_channels,
                     self._volumetric_features_resolution,
                     self._volumetric_features_resolution,
                     self._volumetric_features_resolution),
                   ) * std
                self.register_parameter(
                    '_volumetric_latent_space',
                    torch.nn.parameter.Parameter(p))
                latent_space_memory += p.numel()
        if not self._volumetric_features_time_dependent:
            if self._time_features > 0:
                p = torch.rand((1, self._time_features, input.num_timekeyframes()))
                self.register_parameter(
                    '_time_latent_space',
                    torch.nn.parameter.Parameter(p))
                latent_space_memory += p.numel()
            if self._ensemble_features > 0:
                p = torch.rand((1, self._ensemble_features, input.num_ensembles()))
                self.register_parameter(
                    '_ensemble_latent_space',
                    torch.nn.parameter.Parameter(p))
                latent_space_memory += p.numel()
        print("Latent space memory:", utils.humanbytes(latent_space_memory*4))

    def generalize_to_new_ensembles(self, num_ensembles: int):
        """
        Prepares for generalization-training:
        Replaces the ensemble latent space grid with a new grid for
        the desired number of ensembles.
        :param num_ensembles: the number of ensemble members
        """
        if not hasattr(self, '_volumetric_latent_space_ensemble'):
            raise ValueError("Network wasn't loaded/initialized with ensemble-dependent volumentric latent grids")

        p = torch.randn(
            (num_ensembles, self._ensemble_features,
             self._volumetric_features_resolution,
             self._volumetric_features_resolution,
             self._volumetric_features_resolution),
        ) * self._grid_std
        del self._parameters['_volumetric_latent_space_ensemble']
        self.register_parameter(
            '_volumetric_latent_space_ensemble',
            torch.nn.parameter.Parameter(p))
        return self._volumetric_latent_space_ensemble

    def export_to_pyrenderer(self, opt,
                             grid_encoding, return_grid_encoding_error=False): # pyrenderer.SceneNetwork.LatentGrid.Encoding):
        """
        Exports this network to the pyrenderer TensorCore implementation
        :param opt: the opt dictionary. Used keys:
            layers, activation, ensembles, time_keyframes
        :param grid_encoding:
        :return:
        """
        n = pyrenderer.SceneNetwork()

        # input
        B = self._input_parametrization.get_fourier_feature_matrix()
        n.input.has_direction = self.use_direction()
        if B is not None:
            n.input.set_fourier_matrix_from_tensor(B, self._input_parametrization.is_premultiplied())
        else:
            n.input.disable_fourier_features()
        if self._input_parametrization.has_time():
            if not self._has_volumetric_features and not self._volumetric_features_time_dependent:
                raise ValueError("time input only possible (for now) for time-dependent latent grids")
            n.input.has_time = True
        else:
            n.input.has_time = False

        #output
        n.output.output_mode = pyrenderer.SceneNetwork.OutputParametrization.OutputModeFromString(
            self._outputmode)

        # grid
        encoding_error = 0
        encoding_error_count = 0
        if self._has_volumetric_features:
            if self._volumetric_features_time_dependent:
                time_keyframes = list(range(*map(int, opt['time_keyframes'].split(':'))))
                ensemble_range = list(range(*map(int, opt['ensembles'].split(':'))))
                time_num = len(time_keyframes) if self._time_features > 0 else 0
                ensemble_num = len(ensemble_range) if self._ensemble_features > 0 else 0
                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=time_keyframes[0],
                    time_num=time_num,
                    time_step=time_keyframes[1] - time_keyframes[0] if len(time_keyframes)>1 else 1,
                    ensemble_min=ensemble_range[0],
                    ensemble_num=ensemble_num)

                if self._time_features > 0:
                    grid_time = self._volumetric_latent_space_time
                    assert grid_time.shape[0] == len(time_keyframes)
                    for i in range(len(time_keyframes)):
                        e = grid_info.set_time_grid_from_torch(i, grid_time[i:i + 1], grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                else:
                    assert len(time_keyframes)<=1, "time features disabled, but there were time keyframes in the dataset"

                if self._ensemble_features > 0:
                    grid_ensemble = self._volumetric_latent_space_ensemble
                    assert grid_ensemble.shape[0] == len(ensemble_range)
                    for i in range(len(ensemble_range)):
                        e = grid_info.set_ensemble_grid_from_torch(i, grid_ensemble[i:i+1], grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                else:
                     assert len(ensemble_range)<=1, "ensemble features disabled, but there were ensemble frames in the dataset"

                n.latent_grid = grid_info
            else:
                grid_static = self._volumetric_latent_space
                # save static grid as time grid with one timestep
                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=0, time_num=1, time_step=1,
                    ensemble_min=0, ensemble_num=0)
                e = grid_info.set_time_grid_from_torch(0, grid_static, grid_encoding)
                encoding_error += e
                encoding_error_count += 1
                n.latent_grid = grid_info
            if not n.latent_grid.is_valid():
                raise ValueError("LatentGrid is invalid")

        #hidden
        assert isinstance(self._hidden_layers, InnerNetwork)
        layers = opt['layers']
        activation = opt['activation']
        activationX = activation.split(':')
        activation_param = float(activationX[1]) if len(activationX)>=2 else 1
        activation = pyrenderer.SceneNetwork.Layer.ActivationFromString(activationX[0])
        layer_sizes = list(map(int, layers.split(':')))
        for i, s in enumerate(layer_sizes):
            layer = getattr(self._hidden_layers, 'linear%d' % i)
            assert isinstance(layer, nn.Linear)
            n.add_layer(layer.weight, layer.bias, activation, activation_param)
        last_layer = getattr(self._hidden_layers, 'linear%d'%len(layer_sizes))
        n.add_layer(last_layer.weight, last_layer.bias, pyrenderer.SceneNetwork.Layer.Activation.NONE)

        if not n.valid():
            raise ValueError("Failed to convert network to TensorCores")

        if return_grid_encoding_error:
            return n, encoding_error/encoding_error_count if encoding_error_count>0 else 0
        return n

    def supports_mixed_latent_spaces(self):
        """
        Returns if the network supports mixed latent spaces.
        True -> tf, time, ensemble are torch tensors in the forward method
        False -> tf, time, ensemble are still torch tensors, but only the first entry is used,
            i.e. it is assumed that all latent vectors are identical
        :return: if mixed latent space is supported
        """
        if self._has_meta_network: return False
        if self._volumetric_features_time_dependent: return False
        return True

    def output_mode(self):
        return self._outputmode

    def use_direction(self):
        assert self._use_direction == self._input_parametrization.has_direction()
        return self._use_direction

    def num_time_features(self):
        return self._time_features

    def num_ensemble_features(self):
        return self._ensemble_features

    def num_volumetric_features(self):
        return self._volumetric_features_channels

    def base_input_channels(self):
        return self._base_input_channels

    def total_input_channels(self):
        return self._total_input_channels

    def output_channels(self):
        return self._output_channels

    def _meta_precompute_latent_variables(self, input_data):
        """
        To not store the input data (can't be pickled), precompute the inputs to
        the latent variables already.
        This will later be needed in self.start_epoch()
        :param input_data:
        """
        # query all latent variables
        tfs = []
        ensembles = []
        times = []

        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps('train')
        num_ensembles = input_data.num_ensembles()
        for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
            actual_timestep, actual_ensemble = input_data.compute_actual_time_and_ensemble(
                timestep, ensemble, 'train')
            tfs.append(tf)
            ensembles.append(actual_ensemble)
            times.append(actual_timestep)

        self._meta_tfs = torch.tensor(tfs, device=self._device, dtype=self._dtype)
        self._meta_ensembles = torch.tensor(ensembles, device=self._device, dtype=self._dtype)
        self._meta_times = torch.tensor(times, device=self._device, dtype=self._dtype)

    def start_epoch(self) -> bool:
        """
        Called when an epoch is started.
        This is used to control the pretraining, so that the main script does not
        need to know the details
        :return: true if the optimizer should be reset (i.e. a new phase is entered)
        """
        if self._has_meta_pretraining:
            self._meta_pretrain_current_epoch += 1
            if (self._meta_pretrain_current_epoch > self._meta_pretrain_epoch1):
                print("Pretraining of inner network done, now match the meta-network with the parameters")

                # interpolate latent space
                with torch.no_grad():
                    latent_space = []
                    if self._ensemble_features > 0:
                        ensemble_latent_space = pyrenderer.interp1D(
                            self._ensemble_latent_space,
                            self._meta_ensembles.unsqueeze(1))[..., 0]
                        latent_space.append(ensemble_latent_space)
                    if self._time_features > 0:
                        time_latent_space = pyrenderer.interp1D(
                            self._time_latent_space,
                            self._meta_times.unsqueeze(1))[..., 0]
                        latent_space.append(time_latent_space)
                    z = torch.cat(latent_space, dim=1)

                # finish network
                assert isinstance(self._hidden_layers, InnerNetworkMeta)
                self._hidden_layers.finish_pretraining(z, self._meta_pretrain_epoch2)
                self._has_meta_pretraining = False
                return True
            return False
        else:
            return False # no pretraining -> nothing to do

    def forward(self, x, tf, time, ensemble, mode: str):
        """
        'x' of shape (B,3)
        'tf', 'time', 'ensemble' of shape (B)

        if self.supports_mixed_latent_spaces() == False:
        the only the first entry of 'tf', 'time' and 'ensemble' are used.
        :param x: N,3
        :param tf:
        :param time:
        :param ensemble:
        :return:
        """

        assert mode in ['screen', 'world']

        x2 = [x]
        if self._input_parametrization.has_time():
            x2.append(time.unsqueeze(1))

        if not self.supports_mixed_latent_spaces():
            assert torch.all(ensemble == ensemble[:1])
            assert torch.all(time == time[:1])
            tf = tf[:1]
            ensemble = ensemble[:1]
            time = time[:1]

        latent_space = []
        if self._volumetric_features_time_dependent:
            # copy tf, ensemble, tf to CPU for manual interpolation
            # the slicing to one element above was already done
            # (_volumetric_features_time_dependent implies supports_mixed_latent_spaces()==False)
            #tf = tf.item() #unused at the moment
            if self._time_features > 0:
                time = time.item()
                num_timesteps = self._volumetric_latent_space_time.shape[0]
                time_low = np.clip(int(np.floor(time)), 0, num_timesteps-1)
                time_high = min(time_low+1, num_timesteps-1)
                time_f = time - time_low
                # interpolation in space
                grid_positions = x.unsqueeze(0).unsqueeze(1).unsqueeze(1)  # 1,N,1,1,3
                tmp = F.grid_sample(
                    self._volumetric_latent_space_time[time_low:time_low+1,...],
                    grid_positions * 2 - 1, align_corners=False, padding_mode='border')
                latent_low = tmp[0, :, 0, 0, :].t()
                tmp = F.grid_sample(
                    self._volumetric_latent_space_time[time_high:time_high + 1, ...],
                    grid_positions * 2 - 1, align_corners=False, padding_mode='border')
                latent_high = tmp[0, :, 0, 0, :].t()
                # interpolate in time
                latent_space.append((1-time_f)*latent_low + time_f*latent_high)

            if self._ensemble_features > 0:
                ensemble = ensemble.item()
                num_ensembles = self._volumetric_latent_space_ensemble.shape[0]
                ensemble_low = np.clip(int(np.floor(ensemble)), 0, num_ensembles - 1)
                ensemble_high = min(ensemble_low + 1, num_ensembles - 1)
                ensemble_f = ensemble - ensemble_low
                # interpolation in space
                grid_positions = x.unsqueeze(0).unsqueeze(1).unsqueeze(1)  # 1,N,1,1,3
                tmp = F.grid_sample(
                    self._volumetric_latent_space_ensemble[ensemble_low:ensemble_low + 1, ...],
                    grid_positions * 2 - 1, align_corners=False, padding_mode='border')
                latent_low = tmp[0, :, 0, 0, :].t()
                tmp = F.grid_sample(
                    self._volumetric_latent_space_ensemble[ensemble_high:ensemble_high + 1, ...],
                    grid_positions * 2 - 1, align_corners=False, padding_mode='border')
                latent_high = tmp[0, :, 0, 0, :].t()
                # interpolate in time
                latent_space.append((1 - ensemble_f) * latent_low + ensemble_f * latent_high)

        else:
            if self._ensemble_features > 0:
                ensemble_latent_space = pyrenderer.interp1D(
                    self._ensemble_latent_space,
                    ensemble.unsqueeze(1))[...,0]
                latent_space.append(ensemble_latent_space)
            if self._time_features > 0:
                time_latent_space = pyrenderer.interp1D(
                    self._time_latent_space,
                    time.unsqueeze(1))[...,0]
                latent_space.append(time_latent_space)
            if self._has_volumetric_features:
                input = self._volumetric_latent_space
                grid = x[...,:3].unsqueeze(0).unsqueeze(1).unsqueeze(1) # 1,N,1,1,3
                output = F.grid_sample(input, grid*2-1, align_corners=False, padding_mode='border')
                latent_space.append(output[0,:,0,0,:].t())

        if self._has_meta_network:
            x = torch.cat(x2, dim=1)
            y = self._input_parametrization(x)
            z = torch.cat(latent_space, dim=1)
            y = self._hidden_layers(z, y)
        else:
            x2 = x2 + latent_space
            x = torch.cat(x2, dim=1)
            y = self._input_parametrization(x)
            y = self._hidden_layers(y)
        return self._output_parametrization(y, mode=mode)
