"""
Provides the dataloaders for the training and validation
"""

import argparse
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
import os
from typing import Union, Optional
from itertools import product
import tqdm
import re
import collections
from torch._six import string_classes
import h5py

#debug
import imageio
import matplotlib.pyplot as plt

import common.utils as utils
import pyrenderer
from common.mathparser import BigInteger, BigFloat

from volnet.input_data import TrainingInputData
import volnet.sampling
from volnet.sampling import get_sampled_positions
from volnet.raytracing import Raytracing

_np_str_obj_array_pattern = re.compile(r'[SaUO]')
_cat_collate_err_msg_format = "_cat_collate: batch must contain tensors, numpy arrays, numbers, dicts or lists; found {}"
def _cat_collate(batch):
    r"""
    Puts each data field into a tensor with outer dimension batch size.
    Uses concatenation instead of stacking -> elements are already partially batched
    """

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.cat(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if _np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(_cat_collate_err_msg_format.format(elem.dtype))

            return _cat_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: _cat_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(_cat_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [_cat_collate(samples) for samples in transposed]

    raise TypeError(_cat_collate_err_msg_format.format(elem_type))

class _MCCache:
    def __init__(self, settings_file : Optional[str]):
        """
        Creates the cache.
        If the settings_file is None, the cache is disabled, all queries return None.
        :param settings_file: the settings file as reference for the path
        """
        self._cache_filename = None
        self._cache = None
        self._current_tag = None
        self._current_dset = None
        if settings_file is not None:
            self._cache_filename = os.path.abspath(os.path.splitext(settings_file)[0] + "-cache.hdf5")
            if os.path.exists(self._cache_filename):
                self._cache = h5py.File(self._cache_filename, 'r+')
            else:
                self._cache = h5py.File(self._cache_filename, 'w')

    def query(self, actual_tf, actual_timestep, actual_ensemble, num_views, resolution, num_refine):
        """
        Queries the cached image identified by the parameters
        :return: the image as an np.image of shape B,C,H,W or None if not found
        """
        if self._cache is None: return None # cache disabled
        self._current_tag = f"img_{actual_tf}_{actual_timestep}_{actual_ensemble}_{num_views}_{resolution}_{num_refine}"
        if self._current_tag in self._cache:
            # the images are cached!
            return self._cache[self._current_tag][...]
        return None

    def put(self, data: np.ndarray):
        """
        If the cached images was not found, see query(...), put the newly created images into the cache
        :param data:
        :return:
        """
        if self._current_tag is not None:
            # create dataset
            self._cache.create_dataset(self._current_tag, data=data)
            self._current_tag = None

    def close(self):
        if self._cache is not None:
            self._cache.close()
            self._cache = None

class TrainingData:
    """
    Class to create the dataloaders for training and validation.
    It takes the TrainingInputData instance, renders the images (for screen-space training)
    or computes the sample locations (for world-space training) and assembles the
    dataloaders.

    The dataloaders for training and validation provide different tuples, depending on
    whether world-space or screen-space training is used.
    World: [
        position: float tensor of shape (N,3);
        target: float tensor of shape (N,C) with C=1 for densities, C=4 for colors;
        tf: int tensor of shape (N,) with the index of the TF that was used
        time: int tensor of shape (N,) with the index of the timestep that was used
        ensemble: int tensor of shape (N,) with the index of the ensemble that was used
    ]
    Screen: [
        camera: float tensor of the camera matrices of shape (B,3,3)
        target: float tensor of shape (B,H,W,4) with the rgba-images
           where H=W=train/val:resolution from the ArgumentParser
        tf: int tensor of shape (B,) with the index of the TF that was used
        time: int tensor of shape (B,) with the index of the timestep that was used
        ensemble: int tensor of shape (B,) with the index of the ensemble that was used
        stepsize: float with the stepsize to use
    ]

    """

    SCREEN = "screen"
    WORLD = "world"

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group("Training")
        def add_args(prefix: str, name: str, mode_required):
            group.add_argument(prefix+":mode", type=str, required=mode_required, choices=["screen", "world"], help=f"""
                The basic {name} mode:
                - screen: the network is trained / evaluated via ray tracing from images.
                  This enables the arguments 'views', 'resolution' and 'stepsize'.
                - world: the network is trained in world space: from 3D pos to output.
                  Only later during inference is the network included directly in the renderer.
                """)
            group.add_argument(prefix+":views", type=int, default=8, help=f"""
                [{prefix}:mode=='screen']
                The number of images / views for {name}. The views are sampled
                using the Fibonacci Sphere algorithm
                """)
            group.add_argument(prefix + ":resolution", type=int, default=256, help=f"""
                [{prefix}:mode=='screen']
                The resolution of the images / views for {name} in X and Y direction.
                """)
            group.add_argument(prefix + ":stepsize", type=BigFloat, default=0.01, help=f"""
                [{prefix}:mode=='screen']
                The stepsize for raytracing during {name} in world space.
                Arbitrary math expressions like "1/256" are supported
                """)
            group.add_argument(prefix + ":num_refine", type=int, default=0, help=f"""
                [{prefix}:mode=='screen']
                The number of refinement iterations for monte-carlo traced images
                """)
            group.add_argument(prefix + ":samples", type=BigInteger, default=1024, help=f"""
                [{prefix}:mode=='world']
                The number of sample points for world-space {name}.
                Arbitrary math expressions like "10**4" are supported
                """)
            group.add_argument(prefix + ":sampler", type=str, default="random", choices=["random", "plastic", "halton"], help=f"""
                [{prefix}:mode=='world']
                Specifies the sampling algorithm for world-space {name}.
                """)
            group.add_argument(prefix + ":sampler_importance", type=float, nargs='?', default=None, const=0.1, help=f"""
                [{prefix}:mode=='world']
                If this argument is specified, importance sampling is activated for world-space training.
                The a be the absorption/opacity at the current sample position x and let a_max be the maximal
                possible absorption. Then p=max(pmin, a/a_max) is the probability, that this sample is taken.
                pmin is the minimal probability and specified by this argument. Default: 0.1 -> 10%%
                Furthermore, in this mode, the sampling algorithm is ignored.  
                """)
            group.add_argument(prefix + ":batchsize", type=BigInteger, required=mode_required, help=f"""
                The batch size used during {name}.
                For world-space training, this is the number of points.
                For screen-space training, this is the number of images.
                """)

        add_args('--train', 'training', True)
        group.add_argument('--train:disable_inversion_trick', action='store_true')
        group.add_argument('--val:copy_and_split', type=float, nargs='?', default=None, const=0.2, help="""
            If this argument is specified, the same settings as for training are used for validation.
            The dataset is then split between training and validation using the split factor
            specified by this parameter (default=0.2 -> 20%% for validation).
            """)
        add_args('--val', 'validation', False)
        group.add_argument("--vis:resolution", type=int, default=256, help="""
                        The resolution of the images / views for visualization in X and Y direction.
                        """)
        group.add_argument("--vis:stepsize", type=BigFloat, default=0.005, help="""
                        The stepsize for raytracing during visualization in world space.
                        Arbitrary math expressions like "1/256" are supported
                        """)
        group.add_argument("--vis:num_refine", type=int, default=None, help="""
                        The number of refinement iterations for monte-carlo traced images.
                        Default: use from training data
                        """)
        group.add_argument('--sampler_cache', type=str, default="cache/")
        group.add_argument('--cache_images', action='store_true', help="""
            If specified, rendered images are cached.
            This is especially useful for monte-carlo renderings.""")

        group.add_argument('--rebuild_dataset', type=int, default=-1, help="""
            If provided with a positive value, specifies the number of epochs after which the errors of the networks
            are queried and the dataset is resampled to put more weight on regions with errors.
            Currently only works with world-space training.
            """)
        group.add_argument('--rebuild_gridsize', type=int, default=128,
                           help="The grid size per dimension of the resampling grid")
        group.add_argument('--rebuild_samples_per_voxel', type=int, default=8,
                           help="Number of random samples per voxel grid")
        group.add_argument('--rebuild_importance', type=float, default=None,
                           help="The importance sampling factor for the resampling")
        group.add_argument('--rebuild_force_color', action='store_true',
                           help="Forces loss computation for colors, even if the network predicts densities")

    def __init__(self, opt: dict, dtype, device):
        """
        Initializes the TrainingData with the dictionary obtained from
        the ArgumentParser
        :param opt: the dictionary with the results from the ArgumentParser
        """

        self._opt = opt
        self._dtype = dtype
        self._device = device
        self._copy_and_split = opt['val:copy_and_split']
        self._training_mode = opt['train:mode']
        if self._copy_and_split is None:
            self._validation_mode = opt['val:mode']
        else:
            self._validation_mode = self._training_mode
        self._train_disable_inversion_trick = opt['train:disable_inversion_trick']

        # variables that are available after creating the dataloaders below
        self._train_stepsize = None
        self._train_dataloader = None
        self._val_stepsize = None
        self._val_dataloader = None
        self._vis_dataloader = None

    def train_disable_inversion_trick(self):
        return self._train_disable_inversion_trick

    def is_rebuild_dataset(self):
        return self._opt['rebuild_dataset'] > 0

    def rebuild_dataset_epoch_frequency(self):
        assert self.is_rebuild_dataset()
        return self._opt['rebuild_dataset']

    def _create_dataloader(self,
            dataset: list, batchsize:int, split: Optional[float], shuffle:bool, cat:bool):

        if split is None:
            return torch.utils.data.DataLoader(
                dataset, batch_size=batchsize, shuffle=shuffle, collate_fn=_cat_collate if cat else None), None

        num_entries = len(dataset)
        indices_all = list(range(num_entries))
        np.random.shuffle(indices_all)
        train_val_split = int(len(indices_all) * split)
        indices_val = indices_all[:train_val_split]
        indices_train = indices_all[train_val_split:]
        dataset_val = [dataset[j] for j in indices_val]
        dataset_train = [dataset[j] for j in indices_train]
        return torch.utils.data.DataLoader(dataset_train, batch_size=batchsize, shuffle=shuffle, collate_fn=_cat_collate if cat else None),   \
            torch.utils.data.DataLoader(dataset_val, batch_size=batchsize, shuffle=False, collate_fn=_cat_collate if cat else None)


    def _create_world_dataset(self,
                              prefix: str,
                              input_data: TrainingInputData,
                              network_output_mode: str):
        network_output_mode = network_output_mode.split(':')[0] # trim options
        assert network_output_mode in ["density", "rgbo"]

        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps(prefix)
        num_ensembles = input_data.num_ensembles()
        num_points = self._opt[prefix+":samples"]
        batchsize = self._opt[prefix+":batchsize"]
        importance = self._opt[prefix+":sampler_importance"]

        # make num_points a multiple of batchsize
        num_points = utils.next_multiple(num_points, batchsize)

        # estimate storage size
        N = num_tfs * num_timesteps * num_ensembles * num_points
        num_batches = N//batchsize
        C = 1 if network_output_mode=="density" else 4
        memory = utils.humanbytes(N * (3+C) * 4) # (position (3) + channels (C)) * sizeof(float)
        print(f"Generate world-space dataset for {num_tfs} TFs, {num_timesteps} timesteps, {num_ensembles} ensembles, and {num_points} points.")
        print(f"This gives rise to {N} samples, split over {num_batches} batches with a batch size of {batchsize}.")
        print(f"In total, this requires {memory} for the DataLoader")

        # for more randomness, split each batch into that number of subbatches
        sub_batches = 8 if self._network_supports_mixed_latent_space else 1
        sub_batchsize = max(1, batchsize // sub_batches)
        assert num_points%sub_batchsize==0, "This is ensured by next_multiple()"
        num_splits = num_points // sub_batchsize

        # assemble dataset
        dataset = []
        # For now, use the same sample locations for all ensemble members
        #cache_folder = os.path.abspath(self._opt['sampler_cache'])
        #os.makedirs(cache_folder, exist_ok=True)
        #print("Cache directory for sample locations:", cache_folder)
        cache_folder = None
        #sample_locations = get_sampled_positions(3, num_points, 0, self._opt[prefix+":sampler"], cache_folder)
        #sample_locations_split = np.split(sample_locations, num_splits, axis=0)
        # now compute the target
        sampler_start = 0

        with tqdm.tqdm(num_tfs * num_timesteps * num_ensembles) as iteration_bar:
            for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
                print(f"Load TF {tf}, timestep {timestep}, ensemble {ensemble}")
                image_evaluator, actual_timestep, actual_ensemble = input_data.image_evaluator(tf, timestep, ensemble, mode=prefix)
                volume_interpolation = image_evaluator.volume

                if network_output_mode == "rgbo":
                    ray_evaluator = image_evaluator.ray_evaluator
                    min_density = ray_evaluator.min_density
                    max_density = ray_evaluator.max_density
                    tf_evaluator = ray_evaluator.tf
                else:
                    tf_evaluator = None
                    min_density = 0
                    max_density = 1

                # sample points until we have all
                if importance is None:
                    sample_locations = get_sampled_positions(
                        3, num_points, sampler_start,
                        self._opt[prefix + ":sampler"], cache_folder)
                    sample_locations = torch.from_numpy(sample_locations).to(device=self._device, dtype=self._dtype)
                    densities = volume_interpolation.evaluate(sample_locations)
                    target = densities
                    if network_output_mode == "rgbo":
                        target = tf_evaluator.evaluate(target, min_density, max_density)
                    sampler_start += num_points
                else:
                    seed = 42
                    sample_locations, densities, colors = volume_interpolation.importance_sampling(
                        num_points, tf_evaluator, importance, seed, sampler_start, min_density, max_density, "float")
                    if network_output_mode == "rgbo":
                        target = colors
                    else:
                        target = densities
                    sampler_start += 1

                print(f"Densities: min={densities.min().item()}, max={densities.max().item()}, mean={densities.mean().item()}")
                sample_locations = sample_locations.cpu().numpy()
                target = target.cpu().numpy()

                # split and add to dataset
                sample_locations_split = np.split(sample_locations, num_splits, axis=0)
                target_split = np.split(target, num_splits, axis=0)
                tf_index = np.full((sub_batchsize, ), tf, dtype=np.int32)
                timestep_index = np.full(
                    (sub_batchsize,),
                    input_data.timestep_to_index(actual_timestep),
                    dtype=np.float32)
                ensemble_index = np.full(
                    (sub_batchsize,),
                    input_data.ensemble_to_index(actual_ensemble),
                    dtype=np.float32)
                for j in range(num_splits):
                    dataset.append((
                        sample_locations_split[j], target_split[j],
                        tf_index, timestep_index, ensemble_index))
                iteration_bar.update(1)

        return dataset, sub_batches

    def _create_screen_dataset(self,
                              prefix: str,
                              input_data: TrainingInputData,
                              cache: _MCCache):

        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps(prefix)
        num_ensembles = input_data.num_ensembles()
        views = self._opt[prefix + ":views"]
        resolution = self._opt[prefix + ':resolution']
        stepsize = self._opt[prefix + ':stepsize']
        batchsize = self._opt[prefix + ":batchsize"]
        num_refine = self._opt[prefix + ":num_refine"]
        importance = self._opt[prefix + ":sampler_importance"]
        if not self._network_supports_mixed_latent_space:
            batchsize = 1 # for now, TODO: merge images of the same timestep/tf/ensemble

        # estimate storage size
        N = num_tfs * num_timesteps * num_ensembles * views * resolution * resolution
        num_batches = N // batchsize
        C = 4
        memory = utils.humanbytes(N * (3+3 + C) * 4) # position (3) + direction (3) + channels (C)
        print(
            f"Generate world-space dataset for {num_tfs} TFs, {num_timesteps} timesteps, {num_ensembles} ensembles, "
            f"and {views} views of resolution {resolution}^2.")
        print(f"This gives rise to {N} samples, split over {num_batches} batches with a batch size of {batchsize}.")
        print(f"In total, this requires {memory} for the DataLoader")

        dataset = []

        with tqdm.tqdm(num_tfs * num_timesteps * num_ensembles * views) as iteration_bar:
            iteration_bar.set_description("Render")
            for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
                print(f"Load TF {tf}, timestep {timestep}, ensemble {ensemble}")
                # fetch the image evaluator -> loads the ensemble, timestep and tf
                image_evaluator, actual_timestep, actual_ensemble = input_data.image_evaluator(tf, timestep, ensemble, mode=prefix)
                # compute camera locations
                _camera_center = image_evaluator.camera.center.value
                _camera_center_np = utils.cvector_to_numpy(_camera_center)
                _camera_pitch_cpu, _camera_yaw_cpu = utils.fibonacci_sphere(views)
                _camera_distance = image_evaluator.camera.pitchYawDistance.value.z

                cached_images = cache.query(tf, actual_timestep, actual_ensemble, views, resolution, num_refine)
                if cached_images is None:
                    images_to_cache = [] # rebuild cache
                else:
                    print("Load images from cache")

                for view in range(views):
                    # update camera
                    image_evaluator.camera.pitchYawDistance.value = pyrenderer.double3(
                        _camera_pitch_cpu[view], _camera_yaw_cpu[view], _camera_distance)
                    camera_parameters = image_evaluator.camera.get_parameters()
                    assert camera_parameters.shape[0] == 1 # no batches
                    # render and refine
                    if cached_images is None:
                        target = image_evaluator.render(resolution, resolution)
                        if num_refine > 0:
                            for j in tqdm.trange(num_refine, desc='Refine'):
                                target = image_evaluator.refine(resolution, resolution, target)
                        # tonemapping
                        target = image_evaluator.extract_color(target)
                        target_cpu = target.cpu().numpy()
                        images_to_cache.append(target_cpu)
                    else:
                        target_cpu = cached_images[view,...]
                        target = torch.from_numpy(target_cpu).to(device=self._device)

                    if importance is None:
                        # send whole images and camera to the evaluation
                        input_cpu = camera_parameters.cpu().numpy()
                        N = 1
                    else:
                        # select only specific rays based on the alpha
                        B, H, W, camera_ray_start, camera_ray_dir = \
                            Raytracing.generate_camera_ray(image_evaluator.camera, resolution, resolution, torch.float32)
                        #camera_ray_start.shape = B,H,W,3
                        alpha = target[:,3,:,:].detach()
                        importance_map = alpha * (1 - importance) + importance # B,H,W
                        # Now run the rejection sampling
                        rnd = torch.rand_like(importance_map)
                        mask = importance_map >= rnd
                        def masked_select_4d_cpu(input, mask, chw):
                            B1, H1, W1, C1 = input.shape
                            B2, H2, W2 = mask.shape
                            assert B1==B2; assert H1==H2; assert W1==W2;
                            output = []
                            for c in range(C1):
                                output.append(torch.masked_select(input[:,:,:,c], mask))
                            if chw:
                                # B, C, H, W
                                return torch.stack(output, dim=-1).unsqueeze(-1).unsqueeze(-1).cpu().numpy()
                            else:
                                # B, H, W=, C
                                return torch.stack(output, dim=-1).unsqueeze(1).unsqueeze(1).cpu().numpy()
                        target_cpu = masked_select_4d_cpu(utils.toHWC(target), mask, True)
                        input_cpu = (masked_select_4d_cpu(camera_ray_start, mask, False),
                                     masked_select_4d_cpu(camera_ray_dir, mask, False))
                        N = target_cpu.shape[0]

                    # add to dataset
                    tf_index = np.full((N, 1), tf, dtype=np.int32)
                    timestep_index = np.full((N, 1), input_data.timestep_to_index(actual_timestep), dtype=np.float32)
                    ensemble_index = np.full((N, 1), input_data.ensemble_to_index(actual_ensemble), dtype=np.float32)
                    dataset.append((
                        input_cpu, target_cpu,
                        tf_index, timestep_index, ensemble_index,
                        stepsize))
                    iteration_bar.update(1)

                # save cache
                if cached_images is None:
                    cache.put(np.stack(images_to_cache, axis=0))

        return dataset, batchsize, resolution

    def _create_vis_dataset(self, input_data: TrainingInputData, cache: _MCCache):

        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps('val')
        num_ensembles = input_data.num_ensembles()
        views = 1
        resolution = self._opt['vis:resolution']
        stepsize = self._opt['vis:stepsize']
        num_refine = self._opt['vis:num_refine'] or max(self._opt['train:num_refine'], self._opt['val:num_refine'])
        batchsize = 1

        # estimate storage size
        N = num_tfs * num_timesteps * num_ensembles * views * resolution * resolution
        num_batches = N // batchsize
        C = 4
        memory = utils.humanbytes(N * (3+3 + C) * 4) # position (3) + direction (3) + channels (C)
        print(
            f"Generate world-space dataset for {num_tfs} TFs, {num_timesteps} timesteps, {num_ensembles} ensembles, "
            f"and {views} views of resolution {resolution}^2.")
        print(f"This gives rise to {N} samples, split over {num_batches} batches with a batch size of {batchsize}.")
        print(f"In total, this requires {memory} for the DataLoader")

        dataset = []

        with tqdm.tqdm(num_tfs * num_timesteps * num_ensembles * views) as iteration_bar:
            iteration_bar.set_description("Render")
            for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
                print(f"Load TF {tf}, timestep {timestep}, ensemble {ensemble}")
                # fetch the image evaluator -> loads the ensemble, timestep and tf
                image_evaluator, actual_timestep, actual_ensemble = input_data.image_evaluator(tf, timestep, ensemble, mode='val')
                # reset camera locations
                image_evaluator.camera.pitchYawDistance.value = input_data.default_camera_pitch_yaw_distance()
                camera_parameters = image_evaluator.camera.get_parameters()
                assert camera_parameters.shape[0] == 1 # no batches

                cached_images = cache.query(tf, actual_timestep, actual_ensemble, 1, resolution, num_refine)
                if cached_images is None:
                    # rebuild cache
                    # render and refine
                    img = image_evaluator.render(resolution, resolution)
                    if num_refine > 0:
                        for j in tqdm.trange(num_refine, desc='Refine'):
                            img = image_evaluator.refine(resolution, resolution, img)
                    # tonemapping
                    img = image_evaluator.extract_color(img)
                    # add to dataset
                    img = img.cpu().numpy()
                    cache.put(img)
                    img = img[0,...]
                else:
                    img = cached_images[0, ...]

                tf_index = np.full((1,), tf, dtype=np.int32)
                timestep_index = np.full((1,), input_data.timestep_to_index(actual_timestep), dtype=np.float32)
                ensemble_index = np.full((1,), input_data.ensemble_to_index(actual_ensemble), dtype=np.float32)
                dataset.append((
                    camera_parameters.cpu().numpy()[0], img,
                    tf_index, timestep_index, ensemble_index,
                    stepsize))
                iteration_bar.update(1)

        return dataset, batchsize, resolution

    def create_dataset(self,
                       input_data: TrainingInputData,
                       network_output_mode: str,
                       network_supports_mixed_latent_space: bool):
        """
        Computes the dataset and creates the dataloaders.
        For screen-space training/validation, the images are rendered
        For world-space training/validation, the sample locations are computed.
        :param input_data: the input data
        :param network_output_mode: For world-space evaluation, the output mode of the network.
          Can be "density" or "color" and determines the output channels
          of the target tensor in the dataloader.
        """
        self._network_supports_mixed_latent_space = network_supports_mixed_latent_space
        if input_data.num_tfs()==1 and input_data.num_ensembles()==1 and input_data.num_timesteps('val')==1:
            # if there are no multiple TFs, Ensembles or Timesteps, we can always assume
            # that mixed latent spaces is supported.
            # Currently non-mixed latent spaces is implemented by setting batchsize=1,
            # this decreases the performance in those cases.
            self._network_supports_mixed_latent_space = True

        cache = _MCCache(input_data.settings_file() if self._opt['cache_images'] else None)

        if self._training_mode == "screen":
            dataset1, batchsize1, self._train_image_size = self._create_screen_dataset(
                'train', input_data, cache)
        else: # world
            dataset1, batchsize1 = self._create_world_dataset(
                'train', input_data, network_output_mode)
            self._train_image_size = 0

        if self._copy_and_split is None:
            # only use for training
            self._train_dataloader, _ = self._create_dataloader(
                dataset1, batchsize1, None, True, True)
            # create anew for validation
            if self._validation_mode == "screen":
                dataset2, batchsize2, self._val_image_size = self._create_screen_dataset(
                    'val', input_data, cache)
            else:  # world
                dataset2, batchsize2 = self._create_world_dataset(
                    'val', input_data, network_output_mode)
                self._val_image_size = 0
            self._val_dataloader, _ = self._create_dataloader(
                dataset2, batchsize2, None, False, True)
        else:
            # split into train and validation
            self._train_dataloader, self._val_dataloader = self._create_dataloader(
                dataset1, batchsize1, self._copy_and_split, True, True)
            self._val_image_size = self._train_image_size

        if self._vis_dataloader is None:
            self._vis_dataset, self._vis_batchsize, self._vis_imagesize = self._create_vis_dataset(
                input_data, cache)
            self._vis_dataloader = torch.utils.data.DataLoader(
                self._vis_dataset, batch_size=1, shuffle=False)

        cache.close()

    def rebuild_dataset(
            self, input_data: TrainingInputData, network_output_mode: str,
            network, _time=[1]):
        """
        Rebuilds the world-space dataset while giving more samples to areas where the prediction loss is high
        :param input_data:
        :param network_output_mode:
        :param network:
        :param loss:
        :return:
        """
        assert self._training_mode == "world"
        assert self._copy_and_split is not None # for simplicity
        network_output_mode = network_output_mode.split(':')[0]  # trim options
        assert network_output_mode in ["density", "rgbo"]
        prefix = "train"
        print("Rebuild training data based on the loss distribution")

        # fetch config
        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps(prefix)
        num_ensembles = input_data.num_ensembles()
        num_points_final = self._opt[prefix + ":samples"]
        batchsize = self._opt[prefix + ":batchsize"]
        importance = self._opt["rebuild_importance"] or self._opt[prefix+':sampler_importance']
        grid_size = self._opt['rebuild_gridsize']
        supersampling = self._opt['rebuild_samples_per_voxel']
        force_color = self._opt['rebuild_force_color']
        # make num_points a multiple of batchsize
        num_points_final = utils.next_multiple(num_points_final, batchsize)
        sub_batches_final = 8 if self._network_supports_mixed_latent_space else 1
        sub_batchsize_final = max(1, batchsize // sub_batches_final)
        num_splits_final = num_points_final // sub_batchsize_final

        # loop over ensembles and co
        device = torch.device("cuda")
        num_unbiased_samples = grid_size**3
        sub_batchsize = 64 ** 3
        sub_batches = num_unbiased_samples // sub_batchsize
        assert sub_batches*sub_batchsize == num_unbiased_samples

        dataset = []
        with tqdm.tqdm(num_tfs * num_timesteps * num_ensembles) as iteration_bar:
            for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
                image_evaluator, actual_timestep, actual_ensemble = input_data.image_evaluator(
                    tf, timestep, ensemble, mode=prefix)
                volume_interpolation = image_evaluator.volume

                if network_output_mode == "rgbo" or force_color:
                    ray_evaluator = image_evaluator.ray_evaluator
                    min_density = ray_evaluator.min_density
                    max_density = ray_evaluator.max_density
                    tf_evaluator = ray_evaluator.tf
                else:
                    tf_evaluator = None
                    min_density = 0
                    max_density = 1

                tf_index = np.full((sub_batchsize,), tf, dtype=np.int32)
                timestep_index = np.full(
                    (sub_batchsize,),
                    input_data.timestep_to_index(actual_timestep),
                    dtype=np.float32)
                ensemble_index = np.full(
                    (sub_batchsize,),
                    input_data.ensemble_to_index(actual_ensemble),
                    dtype=np.float32)
                tf_index_gpu = torch.from_numpy(tf_index).to(device=device)
                timestep_index_gpu = torch.from_numpy(timestep_index).to(device=device)
                ensemble_index_gpu = torch.from_numpy(ensemble_index).to(device=device)

                # grid, to be filled with cost
                grid_indices = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size] # XYZ
                assert grid_indices.shape == (3, grid_size, grid_size, grid_size) # 3*X*Y*Z
                # at the end, I want an input position tensor of shape (N,3)
                grid_indices = np.moveaxis(grid_indices, 0, 3) # X*Y*Z*3
                grid_indices_flat = grid_indices.view()
                grid_indices_flat.shape = (grid_size**3, 3) # (XYZ)*3

                # generate unbiased dataset
                with torch.no_grad():
                    loss_grid = torch.zeros((grid_size, grid_size, grid_size),
                                            dtype=torch.float32, device=device)
                    for i in range(supersampling):
                        offsets = np.random.random_sample((grid_size**3, 3))
                        voxel_size = 1 / grid_size
                        sample_positions = (grid_indices_flat + offsets) * voxel_size
                        loss_grid_flat = []
                        for j in range(sub_batches):
                            # sample position
                            sample_positions_part = torch.from_numpy(
                                sample_positions[j*sub_batchsize:(j+1)*sub_batchsize, :])
                            sample_positions_part = sample_positions_part.to(
                                device=device, dtype=torch.float32)
                            # network and loss evaluation
                            predictions = network(
                                sample_positions_part, tf_index_gpu,
                                timestep_index_gpu, ensemble_index_gpu, 'world')
                            if force_color and network_output_mode == "density":
                                predictions = tf_evaluator.evaluate(predictions, min_density, max_density)
                            target = volume_interpolation.evaluate(sample_positions_part)
                            if network_output_mode == "rgbo" or force_color:
                                target = tf_evaluator.evaluate(target, min_density, max_density)
                                absorption_weighting = self._opt['absorption_weighting']
                                l1rgb = F.l1_loss(predictions[..., :3], target[..., :3], reduction='none')
                                l1alpha = F.l1_loss(predictions[..., 3:], target[..., 3:], reduction='none')
                                l1rgb = torch.mean(l1rgb, dim=1, keepdim=True)
                                loss = (l1rgb + absorption_weighting * l1alpha)[:,0]
                            else:
                                loss = F.l1_loss(target, predictions, reduction='none')[:,0]
                            # add to grid
                            loss_grid_flat.append(loss)
                        loss_grid_flat = torch.cat(loss_grid_flat, dim=0) # (XYZ)
                        loss_grid_flat = loss_grid_flat.view((grid_size, grid_size, grid_size))
                        loss_grid += loss_grid_flat

                    # now perform importance sampling using loss_grid as density
                    max_loss = torch.max(loss_grid).item()
                    print("max loss:", max_loss/supersampling)
                    seed = 42
                    sample_locations, densities, colors = volume_interpolation.importance_sampling_with_probability_grid(
                        num_points_final, tf_evaluator,
                        loss_grid, max_loss,
                        importance, seed, _time[0], min_density, max_density)

                ## debug visualize
                #print("Visualize, save to", os.path.abspath('resample-gridSlice%d.png'%_time[0]))
                #fig, ax = plt.subplots(1, 2)
                #ax[0].imshow(loss_grid[:,:,grid_size//2].cpu().numpy())
                #sample_locations_cpu = sample_locations.cpu().numpy() # B,3
                #sample_locations_cpu2 = sample_locations_cpu[(sample_locations_cpu[:,2]>0.47) & (sample_locations_cpu[:,2]<0.53)]
                #ax[1].hist2d(sample_locations_cpu2[:,0], sample_locations_cpu2[:,1], bins=60)
                #fig.savefig('resample-gridSlice%d.png'%_time[0], bbox_inches='tight')
                #plt.close()

                _time[0] += 1
                if network_output_mode == "rgbo":
                    target = colors
                else:
                    target = densities

                print(
                    f"Densities: min={densities.min().item()}, max={densities.max().item()}, mean={densities.mean().item()}")
                sample_locations = sample_locations.cpu().numpy()
                target = target.cpu().numpy()
                sample_locations_split = np.split(sample_locations, num_splits_final, axis=0)
                target_split = np.split(target, num_splits_final, axis=0)
                tf_index = np.full((sub_batchsize_final,), tf, dtype=np.int32)
                timestep_index = np.full(
                    (sub_batchsize_final,),
                    input_data.timestep_to_index(actual_timestep),
                    dtype=np.float32)
                ensemble_index = np.full(
                    (sub_batchsize_final,),
                    input_data.ensemble_to_index(actual_ensemble),
                    dtype=np.float32)
                for j in range(num_splits_final):
                    dataset.append((
                        sample_locations_split[j], target_split[j],
                        tf_index, timestep_index, ensemble_index))

                iteration_bar.update(1)
        # split into training and validation
        self._train_dataloader, self._val_dataloader = self._create_dataloader(
            dataset, sub_batches_final, self._copy_and_split, True, True)


    def training_mode(self):
        return self._training_mode
    def validation_mode(self):
        return self._validation_mode

    def training_image_size(self):
        assert self.training_mode() == 'screen'
        return self._train_image_size
    def validation_image_size(self):
        assert self.validation_mode() == 'screen'
        return self._val_image_size
    def visualization_image_size(self):
        return self._vis_imagesize

    def training_dataloader(self) -> torch.utils.data.DataLoader:
        return self._train_dataloader
    def validation_dataloader(self) -> torch.utils.data.DataLoader:
        return self._val_dataloader
    def visualization_dataloader(self) -> torch.utils.data.DataLoader:
        return self._vis_dataloader