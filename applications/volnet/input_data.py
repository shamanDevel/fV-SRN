"""
Input data for the scene network training
"""

import argparse
import numpy as np
import torch
import os
import itertools
import functools
from typing import Callable, Union

import common.utils as utils
import pyrenderer

class TrainingInputData:

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser, extra_input: Callable=None):
        group = parser.add_argument_group("Input")
        group.add_argument('settings', type=str, help="""
            Settings .json file for the camera, stepsize and initial volume.
        """)
        if extra_input is not None:
            extra_input(group)
        group.add_argument('--tf_directory', type=str, default=None, help="""
            Directory with transfer function files for TF-generalization training.
            If not specified, the TF from the settings is used.
            If specified, this replaces the TF from the settings.
        """)
        group.add_argument('--volume_filenames', type=str, default=None, help="""
            String.format template with keywords "ensemble" and "time" to specify
            volumes for time- and ensemble generalization.
            If not specified, the volume from the settings is used.
            If specified, this replaces the volume from the settings.
            
            Example: "ScalarFlow/sim_{ensemble:06d}/density_{time:06d}.cvol"
        """)
        group.add_argument('--ensembles', type=str, default="0:1", help="""
            Ranges used for the ensemble index. The indices are obtained via
            <code>range(*map(int, {ensembles}.split(':')))</code>
            Example: "0:10:2"
        """)
        group.add_argument('--time_keyframes', type=str, default="0:1", help="""
            Ranges used for the keyframes for time interpolation. 
            At those timesteps, representative vectors are generated, optionally trained,
            and interpolated between timesteps
            The indices are obtained via <code>range(*map(int, {time_keyframes}.split(':')))</code>
            Example: "0:10:2"
        """)
        group.add_argument('--time_train', type=str, default="0:1", help="""
            Ranges used for the timesteps during training.
            If the timesteps don't coincide with the keyframes defined in '--time_keyframes',
            the representative vectors are interpolated.
            The indices are obtained via <code>range(*map(int, {time_keyframes}.split(':')))</code>
            Example: "0:10:2"
        """)
        group.add_argument('--time_val', type=str, default="0:1", help="""
            Ranges used for the timesteps during validation.
            If the timesteps don't coincide with the keyframes defined in '--time_keyframes',
            the representative vectors are interpolated.
            The indices are obtained via <code>range(*map(int, {time_keyframes}.split(':')))</code>
            Example: "0:10:2"
        """)

    _config_file_mapper = None
    @staticmethod
    def set_config_file_mapper(f: Callable[[str], str]):
        """
        Specifies a function that maps config file names from the input settings
        to the actual filename to be loaded.
        This is used to translate from server-side filenames to user-side filenames
        when loading pre-trained models.
        :param f: the callable [str]->str
        """
        TrainingInputData._config_file_mapper = f

    @staticmethod
    def _map_config_file(filename: str):
        if TrainingInputData._config_file_mapper is None:
            return filename
        return TrainingInputData._config_file_mapper(filename)

    def __init__(self, opt: dict, *, check_volumes_exist: bool = True):
        """
        Initializes the TrainingInputData with the dictionary obtained from
        the ArgumentParser
        :param opt: the dictionary with the results from the ArgumentParser
        :param check_volumes_exist: Check that all volumes referenced here
          actually exist (early check before the data loader).
        """

        self._settings_file = opt['settings']
        self._tf_directory = opt['tf_directory']
        self._volume_filenames = opt['volume_filenames']
        self._ensemble_range = map(int, opt['ensembles'].split(':'))
        self._time_keyframes = map(int, opt['time_keyframes'].split(':'))
        self._time_train = map(int, opt['time_train'].split(':'))
        self._time_val = map(int, opt['time_val'].split(':'))

        # load settings
        settings_file = os.path.abspath(self._settings_file)
        settings_file = TrainingInputData._map_config_file(settings_file)
        print("Load settings from", settings_file)
        self._image_evaluator = pyrenderer.load_from_json(settings_file)
        self._default_volume = self._image_evaluator.volume.volume()
        self._default_camera_pitchyawdistance = utils.copy_double3(self._image_evaluator.camera.pitchYawDistance.value)
        self._mipmap_level = self._image_evaluator.volume.mipmap_level()

        # enumerate volumes
        if self._volume_filenames is not None:
            self._ensemble_indices = list(range(*self._ensemble_range))
            self._time_keyframe_indices = list(range(*self._time_keyframes))
            self._time_train_indices = list(range(*self._time_train))
            self._time_val_indices = list(range(*self._time_val))
            # check if filenames exist
            print(f"Specified {len(self._ensemble_indices)*len(self._time_keyframe_indices)} ensemble+time configurations. Now check, if those files exist")
            if check_volumes_exist:
                for ensemble, time in itertools.product(self._ensemble_indices, self._time_keyframe_indices):
                    filename = self._volume_filenames.format(ensemble=ensemble, time=time)
                    if not os.path.exists(filename):
                        print(f"Filename {filename} does not exist!")
                        raise ValueError(f"Filename {filename} does not exist!")
                print("All volumes exist")

        # TODO: enumerate TFs

    def settings_file(self):
        return self._settings_file

    def volume_filenames(self):
        return self._volume_filenames
    def ensemble_indices(self):
        assert self._volume_filenames is not None
        return self._ensemble_indices
    def time_keyframe_indices(self):
        assert self._volume_filenames is not None
        return self._time_keyframe_indices
    def time_train_indices(self):
        assert self._volume_filenames is not None
        return self._time_train_indices
    def time_val_indices(self):
        assert self._volume_filenames is not None
        return self._time_val_indices

    def num_tfs(self):
        return 1 # For now

    def num_timekeyframes(self):
        if self._volume_filenames is None:
            return 1
        else:
            return len(self._time_keyframe_indices)

    def num_timesteps(self, mode:str):
        assert mode in ["train", "val"]
        if self._volume_filenames is None:
            return 1
        else:
            return len(self._time_train_indices) if mode=="train" else len(self._time_val_indices)

    def timestep_to_index(self, timestep: Union[int, float]):
        """
        Converts a timestep to the index into the keyframe array.
        :param timestep: the timestep, int or array-like
        :return: the keyframe index
        """
        if self._volume_filenames is None:
            return 0
        return np.interp(
            timestep,
            self._time_keyframe_indices,
            np.arange(len(self._time_keyframe_indices)))

    def ensemble_to_index(self, ensemble: int):
        """
        Converts a ensemble to the index into the ensemble latent-space array.
        :param ensemble: the ensemble, int or array-like
        :return: the keyframe index
        """
        if self._volume_filenames is None:
            return 0
        return np.interp(
            ensemble,
            self._ensemble_indices,
            np.arange(len(self._ensemble_indices)))

    def num_ensembles(self):
        if self._volume_filenames is None:
            return 1
        else:
            return len(self._ensemble_indices)

    @staticmethod
    @functools.lru_cache(4)
    def _load_volume(filename:str):
        if not os.path.exists(filename):
            raise ValueError("Volume does not exist!", filename)
        return pyrenderer.Volume(filename)

    def image_evaluator(self, tf:int, timestep: int, ensemble:int, mode:str,
                        timestep_and_ensemble_is_actual: bool = False) \
            -> (pyrenderer.IImageEvaluator, int, int):
        """
        Returns the image evaluator to render the scene with the given TF, timestep and ensemble
        :param tf: the TF index in [0, self.num_tfs()-1]
        :param timestep: the timestep index in [0, self.num_timesteps()-1]
        :param ensemble: the ensemble index in [0, self.num_ensembles()-1]
        :param mode: the mode (train or val)
        :param timestep_and_ensemble_is_actual: if false, the timestep and ensemble
          are in the bounds above. If True, the timesteps are as in the actual
          dataset.
        :return: the image evaluator to render the specified scene
        """
        assert mode in ["train", "val"]

        actual_ensemble = ensemble
        actual_timestep = timestep

        # change volume
        if self._volume_filenames is not None:
            volume_interpolation = self._image_evaluator.volume
            if not timestep_and_ensemble_is_actual:
                actual_ensemble = self._ensemble_indices[ensemble]
                actual_timestep = self._time_train_indices[timestep] if mode == 'train' else self._time_val_indices[timestep]
            else:
                actual_ensemble = ensemble
                actual_timestep = timestep
            filename = self._volume_filenames.format(
                ensemble=actual_ensemble,
                time=actual_timestep)
            v = TrainingInputData._load_volume(filename)
            volume_interpolation.setSource(v, self._mipmap_level)

        # TODO: implement multi TFs

        return self._image_evaluator, actual_timestep, actual_ensemble

    def compute_actual_time_and_ensemble(self, timestep: int, ensemble:int, mode:str):
        assert mode in ["train", "val"]

        actual_ensemble = ensemble
        actual_timestep = timestep
        if self._volume_filenames is not None:
            actual_ensemble = self._ensemble_indices[ensemble]
            actual_timestep = self._time_train_indices[timestep] if mode == 'train' else self._time_val_indices[timestep]
        return actual_timestep, actual_ensemble

    def default_image_evaluator(self) -> pyrenderer.IImageEvaluator:
        """
        The default image evaluator from the settings.
        :return:
        """
        self._image_evaluator.volume.setSource(self._default_volume, self._mipmap_level)
        return self._image_evaluator

    def default_camera_pitch_yaw_distance(self):
        return utils.copy_double3(self._default_camera_pitchyawdistance)