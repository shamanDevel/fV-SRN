"""
Inference: loads models from hdf5-files and renders them
"""

import os
import numpy as np
import torch
import torch.nn.functional as F
from typing import Union, List, Any, Optional
import enum
import h5py
import io
import collections
import imageio
from functools import lru_cache
import logging
import subprocess

import common.utils as utils
import pyrenderer

from volnet.network import SceneRepresentationNetwork, InputParametrization
from volnet.input_data import TrainingInputData
from volnet.raytracing import Raytracing

class LoadedModel:
    """
    Class to load trained models from hdf5-checkpoints,
    evaluate them in world and screen space and
    convert them to compiled tensorcore implementations.

    Note: for time-dependent volumes,
    the time-indices are the actual timestep from the underlying dataset.
    That is, the integer values represent actual ground truth data.
    As the latent space variables are usually only defined sparsely,
    the
    """

    class EvaluationMode(enum.Enum):
        TENSORCORES_SHARED = enum.auto()
        TENSORCORES_MIXED = enum.auto()
        PYTORCH32 = enum.auto()
        PYTORCH16 = enum.auto()

    @staticmethod
    def _get_input_data(opt, force_config_file:str, _CACHE=dict()):
        # setup config file mapper
        def mapper(name:str, force_config_file=force_config_file):
            if force_config_file is not None:
                return force_config_file
            #else use from checkpoint
            if os.path.exists(name): return name
            # replace "server" config files with normal config files
            return name.replace('-server.json', '.json')
        TrainingInputData.set_config_file_mapper(mapper)
        # translate volume filenames if trained on the server, evaluated locally
        volume_filenames = opt['volume_filenames']
        if volume_filenames is not None and os.name=='nt':
            base_data_folder = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../..'))
            volume_filenames = volume_filenames.replace("/home/weiss", base_data_folder)
        # filter out options only for TrainingInputData for better caching
        opt2 = {
            'settings': opt['settings'],
            'tf_directory': opt['tf_directory'],
            'volume_filenames': volume_filenames,
            'ensembles': opt['ensembles'],
            'time_keyframes': opt['time_keyframes'],
            'time_train': opt['time_train'],
            'time_val': opt['time_val']
        }
        opt_string = str(opt2)
        d = _CACHE.get(opt_string, None)
        if d is None:
            d = TrainingInputData(opt2, check_volumes_exist=False)
            _CACHE[opt_string] = d
        return d

    @staticmethod
    def setup_config_file_mapper():
        if LoadedModel._config_file_mapper_initialized: return
        def mapper(name:str):
            if os.path.exists(name): return name
            # replace "server" config files with normal config files
            return name.replace('-server.json', '.json')
        TrainingInputData.set_config_file_mapper(mapper)
        LoadedModel._config_file_mapper_initialized = True

    def __init__(self, filename_or_hdf5:Union[str, h5py.File],
                 epoch:int=-1, grid_encoding=None,
                 force_config_file:str=None):
        """
        Loads the network from the filename or directly h5py file.
        :param filename_or_hdf5: the filename
        :param epoch: the epoch to read the weights from
        :param grid_encoding: the grid encoding for TensorCores
        :param force_config_file: if not None, the path to the .json config file
          that is enforced. This overwrites the TF and camera,
          filenames of the volumes are retained.
        """
        if isinstance(filename_or_hdf5, str):
            assert filename_or_hdf5.endswith(".hdf5")
            self._filename = os.path.splitext(os.path.split(filename_or_hdf5)[1])[0]
            print("Load network from", filename_or_hdf5)
            with h5py.File(filename_or_hdf5, 'r') as f:
                self._init_from_hdf5(f, epoch, grid_encoding, force_config_file)
        elif isinstance(filename_or_hdf5, h5py.File):
            self._filename = None
            self._init_from_hdf5(filename_or_hdf5, epoch, grid_encoding, force_config_file)
        else:
            raise ValueError("Unknown argument", filename_or_hdf5)

    def _init_from_hdf5(self, f:h5py.File, epoch:int, grid_encoding, force_config_file:str):
        self._dtype = torch.float32
        self._device = torch.device("cuda")
        self._opt = collections.defaultdict(lambda: None)
        self._opt.update(f.attrs)
        self._input_data = LoadedModel._get_input_data(self._opt, force_config_file)
        self._image_evaluator = self._input_data.default_image_evaluator()
        # self._image_evaluator.selected_channel = pyrenderer.IImageEvaluator.ChannelMode.Color

        total_losses = f['total']
        if total_losses[-1] == 0:
            print("WARNING: Last loss is zero, training most likely didn't finish. Filename: "+f.filename)
        self._training_time = float(f['times'][-1])

        # hack, fix for old networks
        is_new_network = True
        git_hash = self._opt['git'] or ""
        if len(git_hash)>0:
            try:
                exit_code = subprocess.run(["git", "merge-base", "--is-ancestor", "59fc3010267a00d111a16bce591fd6a0e7cd6c8b", git_hash]).returncode
                is_new_network = True if exit_code==0 else False
                print("Based on the git-commit of the checkpoint, it is a %s network"%("new" if is_new_network else "old"))
            except:
                print("unable to check git commit, assume new network architecture")
        InputParametrization.PREMULTIPLY_2_PI = is_new_network


        self._network = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        self._has_direction = self._network.use_direction()
        weights_np = f['weights'][epoch, :]
        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network.to(device=self._device)

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network16 = SceneRepresentationNetwork(self._opt, self._input_data, self._dtype, self._device)
        self._network16.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network16.to(dtype=torch.float16, device=self._device)

        self._volume_grid = self._image_evaluator.volume

        # create tensorcores network
        self._tensorcores_available = False
        if grid_encoding is None:
            grid_encoding = pyrenderer.SceneNetwork.LatentGrid.Float
        try:
            self._scene_network, self._grid_encoding_error = self._network.export_to_pyrenderer(
                self._opt, grid_encoding, return_grid_encoding_error = True)
            self._num_parameters = self._scene_network.num_parameters()

            def to_float3(v):
                return pyrenderer.float3(v.x, v.y, v.z)
            self._scene_network.box_min = to_float3(self._image_evaluator.volume.box_min())
            self._scene_network.box_size = to_float3(self._image_evaluator.volume.box_size())

            self._warps_shared = self._scene_network.compute_max_warps(False)
            self._warps_mixed = self._scene_network.compute_max_warps(True)
            print("Warps shared:", self._warps_shared, ", warps mixed:", self._warps_mixed)

            self._volume_network = pyrenderer.VolumeInterpolationNetwork()
            self._volume_network.set_network(self._scene_network)
            self._tensorcores_available = True
        except Exception as ex:
            print("Unable to load tensor core implementation:", ex)

        print("Loaded, output mode:", self._network.output_mode())

        self._network_output_mode = self._network.output_mode().split(':')[0]  # trim options
        self._raytracing = Raytracing(self._input_data.default_image_evaluator(),
                                      self._network_output_mode, 0.01, 128, 128,
                                      self._dtype, self._device)

        def get_attr_or_None(a):
            return f.attrs[a] if a in f.attrs else None
        self.time_keyframes = get_attr_or_None('time_keyframes')
        self.time_train = get_attr_or_None('time_train')

    def filename(self):
        return self._filename

    def training_time_seconds(self):
        return self._training_time

    def fill_weights(self, weights, epoch:int):
        weights_np = weights[epoch, :]

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)

        weights_bytes = io.BytesIO(weights_np.tobytes())
        self._network16.load_state_dict(
            torch.load(weights_bytes, map_location=self._device), strict=True)
        self._network16.to(dtype=torch.float16, device=self._device)

        self._scene_network = self._network.export_to_pyrenderer(self._opt)

    def is_time_dependent(self):
        """
        Returns true iff the network/data is time- or ensemble-dependent.
        :return:
        """
        return self._input_data.volume_filenames() is not None

    def min_timestep(self):
        """
        If time-dependent, returns the minimal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.time_keyframe_indices()[0]

    def max_timestep(self):
        """
        If time-dependent, returns the maximal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.time_keyframe_indices()[-1]

    def min_ensemble(self):
        """
        If time-dependent, returns the minimal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_indices()[0]

    def max_ensemble(self):
        """
        If time-dependent, returns the maximal timestep index (inclusive)
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_indices()[-1]

    def timestep_interpolation_index(self, timestep: Union[float, int]):
        """
        Given the current timestep (self.min_timestep() <= timestep <= self.max_timestep()),
        returns the interpolation index into the latent space vector or grid
        (in [0, self.get_input_data().num_timekeyframes]).
        :param timestep: the timestep of the data
        :return: the index into the latent space grid
        """
        assert self.is_time_dependent()
        return self._input_data.timestep_to_index(timestep)

    def ensemble_interpolation_index(self, ensemble: Union[float, int]):
        """
        Given the current ensemble (self.min_ensemble() <= ensemble <= self.max_ensemble()),
        returns the interpolation index into the latent space vector or grid
        (in [0, self.get_input_data().num_ensembles()-1]).
        :param ensemble: the ensemble of the data
        :return: the index into the latent space grid
        """
        assert self.is_time_dependent()
        return self._input_data.ensemble_to_index(ensemble)

    def timestep_training_type(self, timestep: int):
        """
        Evaluates how that timestep was used during training.
        Returns a tuple of two booleans
            is_keyframe, is_trained = self.timestep_training_type(timestep)
        Where 'is_keyframe' is true iff there was a keyframe / latent vector at that timestep;
        and 'is_trained' is true iff that timestep was used in the training data
         (either directly via a keyframe or interpolated).
        :param timestep: the timestep to check
        :return: is_keyframe, is_trained
        """
        assert self.is_time_dependent()
        is_keyframe = timestep in self._input_data.time_keyframe_indices()
        is_trained = timestep in self._input_data.time_train_indices()
        return is_keyframe, is_trained

    def save_compiled_network(self, filename):
        if not self._tensorcores_available:
            print("No tensorcores available, can't save")
            return
        self._scene_network.save(filename)

    def warps_mixed(self):
        return self._warps_mixed

    def warps_shared(self):
        return self._warps_shared

    def num_parameters(self):
        return self._num_parameters

    def is_tensorcores_available(self):
        return self._tensorcores_available

    def get_image_evaluator(self):
        return self._input_data.default_image_evaluator()

    def get_input_data(self):
        return self._input_data

    def get_raytracing(self) -> Raytracing:
        return self._raytracing

    def get_network_pytorch(self):
        return self._network, self._network16

    def set_network_pytorch(self, network32, network16):
        self._network = network32
        self._network16 = network16
        self._network_output_mode = self._network.output_mode().split(':')[0]  # trim options
        self._raytracing = Raytracing(self._input_data.default_image_evaluator(),
                                      self._network_output_mode, 0.01, 128, 128,
                                      self._dtype, self._device)

    def get_grid_encoding_error(self):
        return self._grid_encoding_error

    def get_network_tensorcores(self):
        return self._scene_network

    def set_network_tensorcores(self, network):
        self._scene_network = network
        self._volume_network.set_network(self._scene_network)

    def enable_preintegration(self, enabled, convert_to_texture:bool = False):
        re = self._image_evaluator.ray_evaluator
        if convert_to_texture and isinstance(re, pyrenderer.RayEvaluationSteppingDvr):
            re.convert_to_texture_tf()
            print("TF converted to texture")
        tf = re.tf
        if isinstance(tf, pyrenderer.TransferFunctionTexture):
            if enabled:
                tf.preintegration_mode = pyrenderer.TransferFunctionTexture.Preintegrate2D
                print("preintegration enabled")
            else:
                tf.preintegration_mode = pyrenderer.TransferFunctionTexture.Off
        elif enabled:
            print("TF is not a texture, can't use preintegration")

    def set_alpha_early_out(self, enabled:bool):
        re = self._image_evaluator.ray_evaluator
        if isinstance(re, pyrenderer.RayEvaluationSteppingDvr):
            re.early_out = enabled
        else:
            print("Warning, ray evaluator is not an instance of RayEvaluationSteppingDvr, can't set alpha early out")

    def get_default_camera(self) -> torch.Tensor:
        """
        Reference camera as specified in the settings
        """
        image_evaluator = self._input_data.default_image_evaluator()
        _camera_center = image_evaluator.camera.center.value
        image_evaluator.camera.pitchYawDistance.value = self._input_data.default_camera_pitch_yaw_distance()
        camera_parameters = image_evaluator.camera.get_parameters()
        return camera_parameters

    def get_test_cameras(self, N:int) -> List[torch.Tensor]:
        """
        Random cameras based on the fibonacci sphere
        :param N: the number of cameras
        :return:
        """
        image_evaluator = self._input_data.default_image_evaluator()
        _camera_center = image_evaluator.camera.center.value
        _camera_center_np = utils.cvector_to_numpy(_camera_center)
        _camera_pitch_cpu, _camera_yaw_cpu = utils.fibonacci_sphere(N)
        _camera_distance = image_evaluator.camera.pitchYawDistance.value.z
        params = []
        for i in range(N):
            image_evaluator.camera.pitchYawDistance.value = pyrenderer.double3(
                _camera_pitch_cpu[i], _camera_yaw_cpu[i], _camera_distance)
            camera_parameters = image_evaluator.camera.get_parameters().clone()
            params.append(camera_parameters)
        return params

    def get_rotation_cameras(self, N:int) -> List[torch.Tensor]:
        """
        Based on the default setting, rotate around the object
        :param N: num steps for the whole rotation
        :return:
        """
        image_evaluator = self._input_data.default_image_evaluator()
        _camera_center = image_evaluator.camera.center.value
        pyd = self._input_data.default_camera_pitch_yaw_distance()
        pitch = pyd.x
        yaw = pyd.y
        distance = pyd.z
        params = []
        for yaw_offset in np.linspace(0, 2*np.pi, N, endpoint=False):
            image_evaluator.camera.pitchYawDistance.value = pyrenderer.double3(
                pitch, yaw+yaw_offset, distance)
            camera_parameters = image_evaluator.camera.get_parameters().clone()
            params.append(camera_parameters)
        return params

    def get_rotation_camera(self, t:float) -> torch.Tensor:
        """
        Based on the default setting, rotate around the object
        :param t: the time of rotation in [0,1] for a full rotation
        :return: the camera matrix
        """
        image_evaluator = self._input_data.default_image_evaluator()
        _camera_center = image_evaluator.camera.center.value
        pyd = self._input_data.default_camera_pitch_yaw_distance()
        pitch = pyd.x
        yaw = pyd.y
        distance = pyd.z
        params = []
        yaw_offset = 2*np.pi * t
        image_evaluator.camera.set_parameters(torch.empty(0)) # reset to use pitch-yaw-distance again instead of external tensor
        image_evaluator.camera.pitchYawDistance.value = pyrenderer.double3(
            pitch, yaw+yaw_offset, distance)
        camera_parameters = image_evaluator.camera.get_parameters().clone()
        return camera_parameters

    def render_reference(self, camera:Optional[torch.Tensor], width:int, height:int,
                         tf=0, timestep=0, ensemble=0, *,
                         stepsize_world: float = None,
                         timer: pyrenderer.GPUTimer = None,
                         num_refine: int = 0):
        """
        Renders the reference image
        :param camera: the camera tensor, see self.get_default_camera(),
            self.get_test_cameras(), self.get_rotation_cameras()
        :param width: the screen width
        :param height: the screen height
        :param tf: the TF index (currently unused)
        :param timestep: the timestep index, self.min_timestep()<=timestep<=self.max_timestep()
        :param ensemble: the ensemble index, self.min_ensemble()<=ensemble<=self.max_ensemble()
        :param stepsize_world: the stepsize in world coordinates
        :param timer: optinal timer to measure the execution time
        :return: the rendered image of shape (B=1, C=4, H, W)
        """
        image_evaluator, actual_timestep, actual_ensemble = self._input_data.image_evaluator(
            tf, timestep, ensemble, mode='val',
            timestep_and_ensemble_is_actual=True)
        #image_evaluator.volume = self._volume_grid
        if camera is not None:
            image_evaluator.camera.set_parameters(camera)
        if stepsize_world is not None and isinstance(image_evaluator.ray_evaluator, pyrenderer.IRayEvaluationStepping):
            image_evaluator.ray_evaluator.stepsizeIsObjectSpace = False
            image_evaluator.ray_evaluator.stepsize = stepsize_world
        if timer is not None:
            timer.start()
        img = image_evaluator.render(width, height)
        if num_refine>0:
            for _ in range(num_refine):
                img = image_evaluator.refine(width, height, img)
        img = image_evaluator.extract_color(img)
        if timer is not None:
            timer.stop()
        return img.detach()

    def evaluate(self, positions: torch.Tensor, mode:EvaluationMode, tf=0, timestep=0, ensemble=0):
        """
        Evaluates the network in world-space at the given positions
        :param positions: the positions of shape (N,3)
        :param mode: the evaluation mode (TensorCore<->PyTorch)
        :param tf: the TF index (currently unused)
        :param timestep: the timestep index, self.min_timestep()<=timestep<=self.max_timestep()
        :param ensemble: the ensemble index, self.min_ensemble()<=ensemble<=self.max_ensemble()
        :return: the values at the given position of shape (N,C), C=1 for densities, C=4 for color
        """
        assert len(positions.shape)==2
        assert positions.shape[1] == 3

        # convert from actual index to interpolation index
        if self.is_time_dependent():
            timestep = self.timestep_interpolation_index(timestep)
            ensemble = self.ensemble_interpolation_index(ensemble)
        else:
            if timestep!=0 or ensemble!=0:
                logging.warning(f"The current network is not time- or ensemble-dependent, but specified a timestep index (value {timestep}) != 0 or ensemble index (value {ensemble}) != 0.")

        # indices for torch
        dtype_time = torch.float16 if mode == LoadedModel.EvaluationMode.PYTORCH16 else torch.float32
        tf_index = torch.full((positions.shape[0],), tf, dtype=torch.int32, device=self._device)
        time_index = torch.full((positions.shape[0],), timestep, dtype=dtype_time, device=self._device)
        ensemble_index = torch.full((positions.shape[0],), ensemble, dtype=dtype_time, device=self._device)
        network_args = [tf_index, time_index, ensemble_index, 'screen']

        if mode == LoadedModel.EvaluationMode.PYTORCH16:
            pos2 = positions.to(dtype=torch.float16)
            if self._has_direction:
                pos2 = torch.cat((pos2, torch.zeros_like(pos2)), dim=1)
            with torch.no_grad():
                return self._network16(pos2, *network_args).to(dtype=self._dtype)
        elif mode == LoadedModel.EvaluationMode.PYTORCH32:
            pos2 = positions
            if self._has_direction:
                pos2 = torch.cat((pos2, torch.zeros_like(pos2)), dim=1)
            with torch.no_grad():
                return self._network(pos2, *network_args)
        else:
            # TODO: tf, time, ensemble index
            self._volume_network.only_shared_memory = mode == LoadedModel.EvaluationMode.TENSORCORES_SHARED
            old_box_min = self._scene_network.box_min
            old_box_size = self._scene_network.box_size
            self._scene_network.clear_gpu_resources() # so that changing the box has an effect
            self._scene_network.box_min = pyrenderer.float3(0,0,0)
            self._scene_network.box_size = pyrenderer.float3(1,1,1)
            result = self._volume_network.evaluate(positions)
            self._scene_network.box_min = old_box_min
            self._scene_network.box_size = old_box_size
            self._scene_network.clear_gpu_resources()  # for reset
            return result


    def get_max_steps(self, camera:Optional[torch.Tensor], width:int, height:int, stepsize:float):
        """
        Returns the maximal number of steps through the volume
        :param camera:
        :param width:
        :param height:
        :param stepsize:
        :return:
        """
        self._raytracing.set_stepsize(stepsize)
        self._raytracing.set_resolution(width, height)
        return self._raytracing.get_max_steps(camera)

    def render_network(self, camera:Optional[torch.Tensor], width:int, height:int,
                       mode:EvaluationMode, stepsize:float, tf=0, timestep=0, ensemble=0,
                       timer: pyrenderer.GPUTimer = None,
                       num_refine: int = 0):
        """
        Renders an image with the trained scene network
        :param camera: the camera tensor, see self.get_default_camera(),
            self.get_test_cameras(), self.get_rotation_cameras()
        :param width: the screen width
        :param height: the screen height
        :param mode: the evaluation mode (TensorCores<->PyTorch)
        :param stepsize: the stepsize in world coordinates
        :param tf: the TF index (currently unused)
        :param timestep: the timestep index, self.min_timestep()<=timestep<=self.max_timestep()
        :param ensemble: the ensemble index, self.min_ensemble()<=ensemble<=self.max_ensemble()
        :param timer: optional timer to measure the execution time
        :return: the rendered image of shape (B=1, C=4, H, W)
        """

        # convert from actual index to interpolation index
        orig_timestep = timestep
        orig_ensemble = ensemble
        if self.is_time_dependent():
            timestep = self.timestep_interpolation_index(timestep)
            ensemble = self.ensemble_interpolation_index(ensemble)
        else:
            if timestep != 0 or ensemble != 0:
                logging.warning(f"The current network is not time- or ensemble-dependent, but specified a timestep index (value {timestep}) != 0 or ensemble index (value {ensemble}) != 0.")

        # indices for torch
        dtype_time = torch.float16 if mode == LoadedModel.EvaluationMode.PYTORCH16 else torch.float32
        tf_index = torch.full((1, 1), tf, dtype=torch.int32, device=self._device)
        time_index = torch.full((1, 1), timestep, dtype=dtype_time, device=self._device)
        ensemble_index = torch.full((1, 1), ensemble, dtype=dtype_time, device=self._device)
        network_args = [tf_index, time_index, ensemble_index, 'screen']

        with torch.no_grad():
            if timer is not None:
                timer.start()
            if mode==LoadedModel.EvaluationMode.PYTORCH16 or mode==LoadedModel.EvaluationMode.PYTORCH32:
                assert camera is not None # PyTorch-evaluation requires the camera
                is_half = mode==LoadedModel.EvaluationMode.PYTORCH16
                network = self._network16 if is_half else self._network
                self._raytracing.set_stepsize(stepsize)
                self._raytracing.set_resolution(width, height)
                if num_refine>0:
                    # Even though this version is a direct PyTorch port of the monte carlo renderer implemented in CUDA,
                    # it produces slightly less noisy images for the same number of samples (better random numbers?).
                    # Visually equal images are obtained with half the number of samples --> // 2
                    # This just tips the scale away from our implementation and favors the related work.
                    num_samples = max(1, (num_refine+1) * (2**self._image_evaluator.spp_log2) // 2)
                    output = self._raytracing.monte_carlo_trace(
                        network, num_samples, camera, self._image_evaluator.ray_evaluator.tf, is_half, network_args=network_args)
                else:
                    output = self._raytracing.full_trace_forward(
                        network, camera, self._image_evaluator.ray_evaluator.tf, is_half, network_args=network_args)
                if num_refine>0:
                    print("Output max:", torch.max(output[0,:4].reshape(4,-1), dim=1)[0])
                output = self._image_evaluator.extract_color(output)
            else:
                image_evaluator, actual_timestep, actual_ensemble = self._input_data.image_evaluator(
                    tf, int(orig_timestep), orig_ensemble, mode='val',
                    timestep_and_ensemble_is_actual=True)
                ray_evaluator = image_evaluator.ray_evaluator
                # save old
                old_volume = image_evaluator.volume
                if isinstance(ray_evaluator, pyrenderer.IRayEvaluationStepping):
                    old_stepsizeIsObjectSpace = image_evaluator.ray_evaluator.stepsizeIsObjectSpace
                    old_stepsize = image_evaluator.ray_evaluator.stepsize
                # set new
                image_evaluator.volume = self._volume_network
                #print("Old box: min=%s, max=%s"%(old_volume.box_min(), old_volume.box_max()))
                #print("New box: min=%s, max=%s" % (self._volume_network.box_min(), self._volume_network.box_max()))
                if isinstance(ray_evaluator, pyrenderer.IRayEvaluationStepping):
                    ray_evaluator.stepsizeIsObjectSpace = False
                    ray_evaluator.stepsize = stepsize
                self._scene_network.set_time_and_ensemble(orig_timestep, orig_ensemble)
                if camera is not None:
                    image_evaluator.camera.set_parameters(camera)
                self._volume_network.only_shared_memory = mode==LoadedModel.EvaluationMode.TENSORCORES_SHARED
                img = image_evaluator.render(width, height)
                if num_refine > 0:
                    for _ in range(num_refine):
                        img = image_evaluator.refine(width, height, img)
                # restore
                image_evaluator.volume = old_volume # reset
                if isinstance(ray_evaluator, pyrenderer.IRayEvaluationStepping):
                    ray_evaluator.stepsizeIsObjectSpace = old_stepsizeIsObjectSpace
                    ray_evaluator.stepsize = old_stepsize
                # extract color and return
                if num_refine>0:
                    print("Output max:", torch.max(img[0,:4].reshape(4,-1), dim=1)[0])
                img = image_evaluator.extract_color(img)
                output = img.detach()
            if timer is not None:
                timer.stop()
        return output


    @staticmethod
    def convert_image(img):
        out_img = img[0].cpu().detach().numpy()
        out_img *= 255.0
        out_img = out_img.clip(0, 255)
        out_img = np.uint8(out_img)
        out_img = np.moveaxis(out_img, (1, 2, 0), (0, 1, 2))
        return out_img

if __name__ == '__main__':
    #test, example network trained in eval_VolumetricFeatures.py
    ln = LoadedModel('volnet/results/eval_VolumetricFeatures/hdf5/VolumetricLatentSpace-0002-ejecta70-l48x2-fNeRF-G8C16.hdf5')
    num_refine = 0

    tf = 0
    ensemble = ln.min_ensemble() if ln.is_time_dependent() else 0
    time = ln.min_timestep() if ln.is_time_dependent() else 0

    # save
    if ln.is_tensorcores_available():
        output_file = ln.filename() + ".volnet"
        print("Save to", output_file)
        ln.save_compiled_network(output_file)
    else:
        print("No tensorcores available, can't save")

    # reference
    ref_camera = ln.get_default_camera()
    ref = ln.render_reference(ref_camera, 512, 512, tf=tf, timestep=time, ensemble=ensemble, num_refine=num_refine)
    imageio.imwrite('test-reference.png', LoadedModel.convert_image(ref))

    # points
    N = 2048
    positions = torch.rand((N, 3), dtype=ln._dtype, device=ln._device)
    points_torch32 = ln.evaluate(positions, LoadedModel.EvaluationMode.PYTORCH32, tf=tf, timestep=time, ensemble=ensemble)
    points_torch16 = ln.evaluate(positions, LoadedModel.EvaluationMode.PYTORCH16, tf=tf, timestep=time, ensemble=ensemble)
    if ln.is_tensorcores_available():
        points_tc_shared = ln.evaluate(positions, LoadedModel.EvaluationMode.TENSORCORES_SHARED, tf=tf, timestep=time, ensemble=ensemble)
        points_tc_mixed = ln.evaluate(positions, LoadedModel.EvaluationMode.TENSORCORES_MIXED, tf=tf, timestep=time, ensemble=ensemble)
    print("Shape:", points_torch32.shape)
    print("Difference torch32-torch16:", F.mse_loss(points_torch32, points_torch16).item())
    if ln.is_tensorcores_available():
        print("Difference torch16-points_tc_shared:", F.mse_loss(points_torch16, points_tc_shared).item())
        print("Difference torch16-points_tc_mixed:", F.mse_loss(points_torch16, points_tc_mixed).item())
        print("Difference points_tc_shared-points_tc_mixed:", F.mse_loss(points_tc_shared, points_tc_mixed).item())

    # rendering
    stepsize = 0.002

    torch32 = ln.render_network(ref_camera, 512, 512, LoadedModel.EvaluationMode.PYTORCH32, stepsize, tf=tf, timestep=time, ensemble=ensemble, num_refine=num_refine)
    imageio.imwrite('test-torch32.png', LoadedModel.convert_image(torch32))

    torch16 = ln.render_network(ref_camera, 512, 512, LoadedModel.EvaluationMode.PYTORCH16, stepsize, tf=tf, timestep=time, ensemble=ensemble, num_refine=num_refine)
    imageio.imwrite('test-torch16.png', LoadedModel.convert_image(torch16))

    if ln.is_tensorcores_available():
        tensorcores_shared = ln.render_network(
            ref_camera, 512, 512, LoadedModel.EvaluationMode.TENSORCORES_SHARED, stepsize, tf=tf, timestep=time, ensemble=ensemble, num_refine=num_refine)
        imageio.imwrite('test-tensorcores_shared.png', LoadedModel.convert_image(tensorcores_shared))

        tensorcores_mixed = ln.render_network(
            ref_camera, 512, 512, LoadedModel.EvaluationMode.TENSORCORES_MIXED, stepsize, tf=tf, timestep=time, ensemble=ensemble, num_refine=num_refine)
        imageio.imwrite('test-tensorcores_mixed.png', LoadedModel.convert_image(tensorcores_mixed))

    print("Done")
