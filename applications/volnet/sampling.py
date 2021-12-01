import sys

import numpy as np
import torch
import math
import abc
from typing import Union
import os
import tqdm
import logging

import common.utils as utils
import pyrenderer

class ISampler(abc.ABC):
    @abc.abstractmethod
    def sample(self, i: np.ndarray):
        """
        Samples positions in [0,1] with indices i
        :param i: shape (B) of integer indices
        :return: the positions in [0,1]^D with shape (B,D) where the
            dimension D was specified in the constructor
        """
        pass

class RandomSampler(ISampler):
    """
    Simple uniform random sampler.
    Not deterministic! Passing the same indices returns different values
    """
    def __init__(self, d : int):
        self._d = d

    def sample(self, i: np.ndarray):
        B, = i.shape
        return np.random.random_sample((B, self._d))

class PlasticSampler(ISampler):
    """
    Another low-discrepancy sampler based on
    https://stats.stackexchange.com/questions/25528/do-low-discrepancy-sequences-work-in-discrete-spaces
    http://extremelearning.com.au/unreasonable-effectiveness-of-quasirandom-sequences/
    """

    def __init__(self, d : int):
        """
        d: number of dimensions
        """
        self._d = d

        def gamma(d): # Use Newton-Rhapson-Method
            x=1.0000
            for i in range(20):
                x = x-(pow(x,d+1)-x-1)/((d+1)*pow(x,d)-1)
            return x
        g = gamma(d)
        self._alpha = np.zeros(d)
        for j in range(d):
            self._alpha[j] = math.pow(1/g,j+1) % 1
        self._alpha = self._alpha[np.newaxis, :]

    def sample(self, i : np.ndarray):
        z = (0.5 + self._alpha*(i[:,np.newaxis]+1)) % 1
        return z

class HaltonSampler:
    def __init__(self, d : int):
        PRIMES = [2,3,5,7,11,13,17,19,23]
        self._d = d
        self._primes = PRIMES[:d]
        self._radicalInversePermutations = [None] * (max(self._primes)+1)
        for i in self._primes:
            self._radicalInversePermutations[i] = np.arange(i)
            np.random.shuffle(self._radicalInversePermutations[i])

    def _radicalInverse(self, a : int, base : int):
        invBase = 1.0 / base
        reversedDigits = 0
        invBaseN = 1.0
        perm = self._radicalInversePermutations[base]
        while a>0:
            next = a // base
            digit = a - next * base
            reversedDigits = reversedDigits * base + perm[digit]
            invBaseN *= invBase
            a = next
        return min(invBaseN * (reversedDigits + \
            invBase * perm[0] / (1-invBase)), 1 - sys.float_info.epsilon)

    def sample(self, i: np.ndarray):
        # TODO: vectorize
        B, = i.shape
        ret = np.empty((B,self._d), dtype=np.float32)
        for a in range(B):
            for b in range(self._d):
                ret[a,b] = self._radicalInverse(i[a], self._primes[b])
        return ret

def get_sampled_positions(dimension: int, num_samples: int, start_index: int,
                          sampler: str, cache_folder: str = None):
    """
    Computes sampled positions, the positions are in [0,1]^D with shape (B,D)
    :param dimension: the dimension 'D'
    :param num_samples: the batch size / number of samples 'B'
    :param start_index: the start index for deterministic samplers
    :param sampler: the sampler type, 'plastic' or 'halton'
    :param cache_folder: the cache folder, must exist. If 'None', no caching
    :return: the positions in [0,1]^D with shape (B,D) as numpy array
    """
    if cache_folder is not None:
        assert os.path.isdir(cache_folder)
    assert sampler in ['random', 'plastic', 'halton']

    if start_index>0 and cache_folder is not None:
        logging.warning("start_index>0, but caching is enabled. Ignoring start_index to load from cache")
        start_index = 0

    # parse cache folder
    best_file = None
    min_num = None

    if cache_folder is not None:
        prefix = "%s%d-"%(sampler, dimension)
        suffix = ".npy"

        for f in os.listdir(cache_folder):
            if os.path.isfile(os.path.join(cache_folder, f)) and \
                f.startswith(prefix) and f.endswith(suffix):
                num = int(f[len(prefix):-len('.npy')])
                if num >= num_samples:
                    if min_num is None or min_num>num:
                        min_num = num
                        best_file = f

    if best_file is not None:
        print("Load",num_samples,"samples from cache file", best_file)
        content = np.load(os.path.join(cache_folder, best_file), allow_pickle=False)
        return content[:num_samples,:]
    else:
        print("Generate",num_samples,"new samples")
        if sampler == 'plastic':
            s = PlasticSampler(dimension)
        elif sampler == "halton":
            s = HaltonSampler(dimension)
        elif sampler == 'random':
            s = RandomSampler(dimension)
        else:
            raise ValueError("Unknown sampler %s"%sampler)

        batch_size = 2**14
        batches = int(math.ceil(num_samples/batch_size))
        content = np.empty((num_samples, dimension), dtype=np.float32)
        with tqdm.trange(batches) as t:
            for batch in t:
                start = batch*batch_size
                end = min(num_samples, start+batch_size)
                indices = np.arange(start, end, dtype=np.int32) + start_index
                content[start:end, :] = s.sample(indices)

        if cache_folder is not None:
            output_file = os.path.join(cache_folder, prefix+str(num_samples)+suffix)
            print("Done, save to cache", output_file)
            np.save(output_file, content)

        return content

def get_sampled_positions_importance(
    num_samples: int, sampler: str,
    volume_density: torch.Tensor, tf: pyrenderer.ITransferFunction,
    weight_uniform: float, weight_density_gradient: float,
	weight_opacity: float, weight_opacity_gradient: float,
    cache_folder: str) -> torch.Tensor:
    """
    Computes sampled positions, the positions are in [0,1]^3 with shape (B,3)
    :param num_samples: the batch size / number of samples 'B'
    :param sampler: the sampler type, 'plastic' or 'halton'
    :param volume_density:
    :param tf:
    :param tf_mode:
    :param weight_uniform:
    :param weight_density_gradient:
    :param weight_opacity:
    :param weight_opacity_gradient:
    :param cache_folder: the cache folder, must exist
    :return: the positions in [0,1]^3 with shape (B, 3) as numpy tensor
    """
    sample_locations = get_sampled_positions(3, num_samples, sampler, cache_folder)
    results = []
    max_batch = 256**3
    num_sections = max(1, sample_locations.shape[0] // max_batch)
    for batch in np.array_split(sample_locations, num_sections, axis=0):
        sample_locations = torch.from_numpy(batch).to(device=volume_density.device, dtype=renderer_dtype_torch)
        seed = np.random.randint(0x7fffffff)
        mask = pyrenderer.sample_importance(
            sample_locations, volume_density, tf, tf_mode,
            weight_uniform, weight_density_gradient, weight_opacity, weight_opacity_gradient, seed)
        result = torch.stack([torch.masked_select(sample_locations[:,i], mask) for i in range(3)], dim=1).cpu().numpy()
        results.append(result)
    result = np.concatenate(results, axis=0)
    print(result.shape[0], "positions sampled")
    return result

def __test_importance_sampling():
    # settings
    print("Load settings ...")
    settingsFile = os.path.abspath(os.path.join(os.path.split(__file__)[0], '../../config-files/skull4gauss.json'))
    from diffdvr import Settings
    from tempfile import TemporaryDirectory
    s = Settings(settingsFile)
    reference_volume = s.load_dataset()
    reference_volume.copy_to_gpu()
    reference_volume_data = reference_volume.getDataGpu(0)
    device = reference_volume_data.device

    # tf
    tf_mode = pyrenderer.TFMode.Gaussian
    tf = s.get_gaussian_tensor().to(device=device, dtype=renderer_dtype_torch)

    print("Sample points")
    N = 1024 * 1024
    sampler = "halton"
    with TemporaryDirectory() as cache_dir:
        def write(weightUniform, weightDensityGradient, weightOpacity, weightOpacityGradient):
            outputFile = os.path.abspath(os.path.join(os.path.split(__file__)[0],
                                      '../../matlab/skull4gauss-points-%d-%d-%d-%d-%d.txt' % (
                                      N, weightUniform, weightDensityGradient, weightOpacity,
                                      weightOpacityGradient)))
            sample_locations = get_sampled_positions_importance(
                N, sampler, reference_volume_data, tf, tf_mode,
                weightUniform, weightDensityGradient, weightOpacity,
                weightOpacityGradient, cache_dir)
            print("Locations generated, now save to file")
            sample_locations = sample_locations.cpu().numpy()
            print(sample_locations.shape)
            np.savetxt(outputFile, sample_locations, '%.5f')

        write(1,0,0,0)
        write(0,1,0,0)
        write(0,0,1,0)
        write(0,0,0,1)
        write(1,0,100,0)

    print("Done")

if __name__ == '__main__':
    __test_importance_sampling()

