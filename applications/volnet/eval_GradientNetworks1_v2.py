import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import sys
import os
import subprocess
import imageio
import shutil
import json
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as patches
import matplotlib.cbook
import seaborn as sns
import itertools
from typing import NamedTuple, Tuple, List, Optional, Any, Union

import common.utils as utils
import pyrenderer
from volnet.inference import LoadedModel
from volnet.network_gradients import NetworkGradientTransformer
from losses.lossbuilder import LossBuilder
from volnet.sampling import PlasticSampler

BASE_PATH = 'volnet/results/eval_GradientNetworks1_v2'

BEST_GRID_RESOLUTION = 32
BEST_GRID_CHANNELS = 16 #32
BEST_NETWORK_LAYERS = 4 #6
BEST_NETWORK_CHANNELS = 32
BEST_ACTIVATION = "SnakeAlt:1"
BEST_FOURIER_STD = -1 # NERF
BEST_FOURIER_COUNT = 14 # to fit within 32 channels

DEFAULT_NUM_SAMPLES = "512**3"
DEFAULT_NUM_EPOCHS = 300
DEFAULT_STEPSIZE = 1 / 1024 #1 / 512

GRADIENT_WEIGHT_RANGE_MAX = -2
GRADIENT_WEIGHT_RANGE_MIN = -10
GRADIENT_WEIGHT_SCALE = 0.5
GRADIENT_WEIGHT_DEFAULT_VALUE = -6

# only use cosine similarity on gradients longer than this value
EVAL_WORLD_NUM_POINTS = 256**3 #512**3
EVAL_SCREEN_SIZE = 1024
EVAL_LENGTH_THRESHOLDS = [0.0, 0.01, 0.1, 1.0]
EVAL_LENGTH_THRESHOLDS_IDX_PLOT = 1
EVAL_SCREEN_FD_SCALES = [(1, '*1', '_x1')]
EVAL_WORLD_FD_SCALES = EVAL_SCREEN_FD_SCALES
EVAL_WORLD_AD_SCALES = [(4, '*4')]

class Config(NamedTuple):
    name: str
    human_name: str
    settings: str
    grid_size: int
    overwrite_layers: Optional[int] = None
    overwrite_samples: Optional[str] = None
    overwrite_epochs: Optional[int] = None
    synthetic: bool = False
    use_in_teaser: bool = False

configX = [
    Config(
        name = "Blobby",
        human_name = "Blobby",
        settings = "config-files/implicit-Blobby.json",
        grid_size = 128,
        synthetic = True
    ),
    Config(
        name = "MarschnerLobb",
        human_name = "Marschner~Lobb",
        settings = "config-files/implicit-MarschnerLobb.json",
        grid_size = 256,
        synthetic = True
    ),
    Config(
        name = "Jet",
        human_name = "Jet",
        settings = "config-files/LuBerger-Jet-v3-shaded.json",
        grid_size = 512,
        use_in_teaser = True
    ),
    Config(
        name = "Ejecta1024",
        human_name = "Ejecta",
        settings = "config-files/ejecta1024-v7-shaded.json",
        grid_size = 1024,
        overwrite_samples = "1024**3",
        overwrite_epochs=100
    ),
]

def main():
    cfgs = []
    for config in configX:
        print("\n==========================================")
        print(config.name)
        print("==========================================")

        train(config)
        statistics_file = eval(config)
        cfgs.append((config, statistics_file))

    print("\n==========================================")
    print("MAKE PLOTS")
    print("==========================================")
    make_plots(cfgs)

def _gradient_weight(index: int):
    """
    Converts from the weight index in [GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX]
    to the actual gradient weight in [0,1]
    """
    return np.tanh(GRADIENT_WEIGHT_SCALE*index)*0.5 + 0.5

def _run_name(config: Config, gradient_weight: Optional[int]):
    """
    Returns the name of the run for the given config and gradient weight index.
    If the gradient weight index is None, the network is trained without gradients.
    :param config:
    :param gradient_weight:
    :return: the run name
    """
    if gradient_weight is None:
        return config.name + "-NoGradient"
    else:
        return config.name + "-Gradient%+02d"%gradient_weight

def train(config: Config):
    best_network_layers = config.overwrite_layers or BEST_NETWORK_LAYERS
    training_samples = config.overwrite_samples or DEFAULT_NUM_SAMPLES
    epochs = config.overwrite_epochs or DEFAULT_NUM_EPOCHS

    common_args = [
        sys.executable, "volnet/train_volnet.py",
        config.settings,
        "--train:mode", "world",
        "--train:samples", training_samples,
        '--rebuild_dataset', '51',
        '--rebuild_importance', '0.1',
        "--val:copy_and_split",
        "--layers", ':'.join([str(BEST_NETWORK_CHANNELS)] * (best_network_layers - 1)), # -1 because last layer is implicit
        "--train:batchsize", "64*64*128",
        "--activation", BEST_ACTIVATION,
        '--fouriercount', str(BEST_FOURIER_COUNT),
        '--fourierstd', str(BEST_FOURIER_STD),
        '--volumetric_features_channels', str(BEST_GRID_CHANNELS),
        '--volumetric_features_resolution', str(BEST_GRID_RESOLUTION),
        "-l1", "1",
        '-lr', '0.01',
        "--lr_step", "100",
        "-i", str(epochs),
        "--logdir", BASE_PATH + '/log',
        "--modeldir", BASE_PATH + '/model',
        "--hdf5dir", BASE_PATH + '/hdf5',
        '--save_frequency', '20',
    ]

    def args_no_grad():
        return [
            "--outputmode", "density:direct",
            "--lossmode", "density",
        ]
    def args_with_grad(weight_index: int):
        return [
            "--outputmode", "densitygrad:direct",
            "--lossmode", "densitygrad",
            "--gradient_weighting", str(_gradient_weight(weight_index)),
            "--gradient_l1", "0",
            "--gradient_l2", "1",
        ]

    def run(args, filename):
        args2 = args + ["--name", filename]
        if os.path.exists(os.path.join(BASE_PATH, 'hdf5', filename+".hdf5")):
            print("Skipping", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args2, check=True)

    run(common_args + args_no_grad(), _run_name(config, None))
    for i in range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX+1):
        run(common_args + args_with_grad(i), _run_name(config, i))

class NetworkWrapperExtractDensity(nn.Module):
    """
    Wraps a densitygrad-network and returns only the density
    """
    def __init__(self, net: nn.Module):
        super().__init__()
        self._net = net

    def forward(self, x, *args, **kwargs):
        y = self._net(x, *args, **kwargs)
        return y[...,:1]

class VolumeEvaluation(nn.Module):
    def __init__(self, vol: pyrenderer.IVolumeInterpolation):
        super().__init__()
        self._vol = vol
    def forward(self, x, *args, **kwargs):
        return self._vol.evaluate(x)

class VolumeEvaluationWithGradient(nn.Module):
    def __init__(self, vol: pyrenderer.IVolumeInterpolation):
        super().__init__()
        self._vol = vol
    def forward(self, x, *args, **kwargs):
        densities, gradients = self._vol.evaluate_with_gradients(x)
        return torch.cat((densities, gradients), dim=-1)
    def use_direction(self):
        return False


def _eval_world(interp_or_net: Union[pyrenderer.VolumeInterpolationNetwork, torch.nn.Module],
                dataloader: torch.utils.data.DataLoader,
                network_args: Any = None,
                no_gradients: bool = False,
                input16bit: bool = False):
    device = torch.device('cuda')
    dtype32 = torch.float32
    dtype16 = torch.float16

    density_l1 = None
    density_l2 = None
    gradient_l1 = None
    gradient_l2 = None
    gradient_length_l1 = None
    gradient_cosine_simX = [None] * len(EVAL_LENGTH_THRESHOLDS)
    weights = None
    time_seconds = 0
    timer = pyrenderer.GPUTimer()
    def append(out, v):
        v = v.cpu().numpy()
        if out is None: return v
        return np.concatenate((out, v), axis=0)

    with torch.no_grad():
        #if not isinstance(interp_or_net, torch.nn.Module):
        #    scene_network = interp_or_net.current_network()
        #    old_box_min = scene_network.box_min
        #    old_box_size = scene_network.box_size
        #    scene_network.clear_gpu_resources()  # so that changing the box has an effect
        #    scene_network.box_min = pyrenderer.float3(0, 0, 0)
        #    scene_network.box_size = pyrenderer.float3(1, 1, 1)

        warmup = True
        for locations_gt, densities_gt, gradients_gt, opacities_gt in tqdm.tqdm(dataloader):
            locations_gt = locations_gt[0].to(device=device)
            densities_gt = densities_gt[0].to(device=device)
            gradients_gt = gradients_gt[0].to(device=device)
            opacities_gt = opacities_gt[0].to(device=device)

            if isinstance(interp_or_net, torch.nn.Module):
                # Native Pytorch
                if warmup:
                    if input16bit:
                        prediction = interp_or_net(locations_gt.to(dtype=torch.float16), *network_args)
                    else:
                        prediction = interp_or_net(locations_gt, *network_args)
                    warmup = False
                timer.start()
                if input16bit:
                    prediction = interp_or_net(locations_gt.to(dtype=torch.float16), *network_args)
                else:
                    prediction = interp_or_net(locations_gt, *network_args)
                timer.stop()
                time_seconds += timer.elapsed_milliseconds()/1000.0
                densities_pred = prediction[:,:1]
                if not no_gradients:
                    gradients_pred = prediction[:,1:]
                else:
                    gradients_pred = None
            else:
                # Custom TensorCore Implementation
                if warmup:
                    if no_gradients:
                        densities_pred = interp_or_net.evaluate(locations_gt)
                    else:
                        densities_pred, gradients_pred = interp_or_net.evaluate_with_gradients(locations_gt)
                    warmup = False
                timer.start()
                if no_gradients:
                    densities_pred = interp_or_net.evaluate(locations_gt)
                    gradients_pred = None
                else:
                    densities_pred, gradients_pred = interp_or_net.evaluate_with_gradients(locations_gt)
                timer.stop()
                time_seconds += timer.elapsed_milliseconds() / 1000.0

            density_l1 = append(density_l1, torch.abs(densities_gt-densities_pred)[:,0])
            density_l2 = append(density_l2, F.mse_loss(densities_gt, densities_pred, reduction='none')[:,0])

            if not no_gradients:
                weights = append(weights, opacities_gt[:,0])
                gradient_l1 = append(gradient_l1,
                                     torch.mean(torch.abs(gradients_gt - gradients_pred), dim=1))
                gradient_l2 = append(gradient_l2,
                                     torch.mean(F.mse_loss(gradients_gt, gradients_pred, reduction='none'), dim=1))
                len_gt = torch.linalg.norm(gradients_gt, dim=1, keepdim=True)
                len_pred = torch.linalg.norm(gradients_pred, dim=1, keepdim=True)
                gradient_length_l1 = append(gradient_length_l1, torch.abs(len_gt - len_pred)[:,0])
                len_gt = torch.clip(len_gt, min=1e-5)
                len_pred = torch.clip(len_pred, min=1e-5)
                N = gradients_gt.shape[0]
                cosine_sim = torch.bmm((gradients_gt / len_gt).reshape(N, 1, 3),
                                       (gradients_pred / len_pred).reshape(N, 3, 1))
                cosine_sim = cosine_sim[:,0,0]
                len_gt = len_gt[:,0]

                for i in range(len(EVAL_LENGTH_THRESHOLDS)):
                    length_mask = len_gt >= EVAL_LENGTH_THRESHOLDS[i]
                    cosine_sim_filtered = torch.masked_select(cosine_sim, length_mask)
                    gradient_cosine_simX[i] = append(gradient_cosine_simX[i], cosine_sim_filtered)

        #if not isinstance(interp_or_net, torch.nn.Module):
        #    scene_network = interp_or_net.current_network()
        #    scene_network.box_min = old_box_min
        #    scene_network.box_size = old_box_size
        #    scene_network.clear_gpu_resources()  # for reset

    def extract_stat(v, weights=None):
        # create histogram
        frequencies, bin_edges = np.histogram(v, bins=50, weights=weights)
        # create boxplot stats
        bxpstats = matplotlib.cbook.boxplot_stats(v)
        for d in bxpstats:
            d['fliers'] = list() # delete fliers (too big)

        # fill dictionary
        avg = np.average(v, weights=weights)
        if weights is None:
            std = np.std(v)
        else:
            std = np.sqrt(np.average((v-avg)**2, weights=weights))
        return {
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'mean': float(avg),
            'median': float(np.median(v)),
            'std': float(std),
            'histogram': {"frequencies": list(map(int, frequencies)), "bin_edges": list(map(float, bin_edges))},
            'bxpstats': bxpstats
        }
    if no_gradients:
        return {
            'density_l1': extract_stat(density_l1),
            'density_l2': extract_stat(density_l2),
            'total_time_seconds': float(time_seconds)
        }
    else:
        return {
            'density_l1': extract_stat(density_l1),
            'density_l2': extract_stat(density_l2),
            'gradient_l1': extract_stat(gradient_l1),
            'gradient_l2': extract_stat(gradient_l2),
            'length_l1': extract_stat(gradient_length_l1),
            'cosine_similarity': [
                {'threshold': EVAL_LENGTH_THRESHOLDS[i], 'data': extract_stat(gradient_cosine_simX[i])}
                for i in range(len(EVAL_LENGTH_THRESHOLDS))
            ],
            'gradient_l1_weighted': extract_stat(gradient_l1, weights=weights),
            'gradient_l2_weighted': extract_stat(gradient_l2, weights=weights),
            'length_l1_weighted': extract_stat(gradient_length_l1, weights=weights),
            'cosine_similarity_weighted': [
                {'threshold': 0.0, 'data': extract_stat(gradient_cosine_simX[0], weights=weights)}
            ],
            'total_time_seconds': float(time_seconds)
        }

def eval(config: Config):
    """
    Evaluates the networks in world- and screen-space
    :param config:
    :return:
    """

    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats-%s.json' % config.name)
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    timer = pyrenderer.GPUTimer()
    device = torch.device('cuda')
    dtype = torch.float32

    #world
    num_points = EVAL_WORLD_NUM_POINTS #256**3 #512**3
    batch_size = min(EVAL_WORLD_NUM_POINTS, 128**3)
    num_batches = num_points // batch_size

    #screen
    width = EVAL_SCREEN_SIZE
    height = EVAL_SCREEN_SIZE
    stepsize = DEFAULT_STEPSIZE
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    grid_encoding = pyrenderer.SceneNetwork.LatentGrid.ByteLinear #.Float
    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    output_stats = {
        "name": config.name,
        "settings": config.settings,
    }

    # Load networks
    torch.cuda.empty_cache()
    def load_and_save(i: Optional[int]):
        filename = _run_name(config, i)
        filename = os.path.abspath(os.path.join(BASE_PATH, 'hdf5', filename+".hdf5"))
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            raise ValueError("File not found: "+filename)
        try:
            ln = LoadedModel(filename, force_config_file=config.settings,
                             grid_encoding=grid_encoding)
            volnet_filename = filename.replace('.hdf5', '.volnet')
            ln.save_compiled_network(volnet_filename)
            volnet_filesize = os.path.getsize(volnet_filename)
            return ln, filename, volnet_filesize
        except Exception as e:
            print("Unable to load '%s':" % filename, e)
            raise ValueError("Unable to load '%s': %s" % (filename, e))
    lns = dict()
    lns['nograd'] = load_and_save(None)
    for i in range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX + 1):
        lns[i] = load_and_save(i)
    base_ln: LoadedModel = lns['nograd'][0]
    network_fd = NetworkGradientTransformer.finite_differences(
        base_ln.get_network_pytorch()[0], 1/config.grid_size)
    network_ad = NetworkGradientTransformer.autodiff(
        base_ln.get_network_pytorch()[0])

    # EVALUATE SCREEN
    print("-- EVALUATE SCREEN SPACE --")
    image_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(image_folder, exist_ok=True)

    camera = base_ln.get_default_camera()
    reference_image = base_ln.render_reference(
        camera, width, height, timer=None, stepsize_world=stepsize,
        channel=pyrenderer.IImageEvaluator.Color) # warmup
    base_ln.render_reference(
        camera, width, height, timer=timer, stepsize_world=stepsize,
        channel=pyrenderer.IImageEvaluator.Color)  # timing
    reference_feature = base_ln.get_image_evaluator().volume.volume().get_feature(0)
    channels = reference_feature.channels()
    resolution = reference_feature.base_resolution()
    bytes_per_voxel = pyrenderer.Volume.bytes_per_type(reference_feature.type())
    reference_volume_size = bytes_per_voxel * channels * \
                    resolution.x * resolution.y * resolution.z
    output_stats['reference_volume_size'] = reference_volume_size
    screen_time_reference = timer.elapsed_milliseconds()/1000.0
    imageio.imwrite(
        os.path.join(image_folder, '%s-color-reference.png' % config.name),
        LoadedModel.convert_image(reference_image))
    reference_normal_image = base_ln.render_reference(
        camera, width, height, timer=None, stepsize_world=stepsize,
        channel=pyrenderer.IImageEvaluator.Normal)
    imageio.imwrite(
        os.path.join(image_folder, '%s-normal-reference.png' % config.name),
        LoadedModel.convert_image(reference_normal_image))
    output_stats_screen = {}

    def _eval_screen(ln, name, mode, override_network=None):
        with torch.no_grad():
            ln.render_network(
                camera, width, height, mode, stepsize,
                override_network=override_network, timer=None,
                channel=pyrenderer.IImageEvaluator.Color) # warmup
            current_image = ln.render_network(
                camera, width, height, mode, stepsize,
                override_network=override_network, timer=timer,
                channel=pyrenderer.IImageEvaluator.Color) # actual rendering
            imgname = os.path.join(image_folder, '%s-color-%s.png' % (config.name, name))
            imageio.imwrite(
                imgname,
                LoadedModel.convert_image(current_image))
            # normal image
            normal_image = ln.render_network(
                camera, width, height, mode, stepsize,
                override_network=override_network, timer=None,
                channel=pyrenderer.IImageEvaluator.Normal)
            normal_imgname = os.path.join(image_folder, '%s-normal-%s.png' % (config.name, name))
            imageio.imwrite(
                normal_imgname,
                LoadedModel.convert_image(normal_image))
            # return stats
            return {
                "time_seconds": timer.elapsed_milliseconds()/1000.0,
                "ssim-color": ssim_loss(current_image, reference_image).item(),
                "lpips-color": lpips_loss(current_image, reference_image).item(),
                "ssim-normal": ssim_loss(normal_image, reference_normal_image).item(),
                "lpips-normal": lpips_loss(normal_image, reference_normal_image).item(),
                'color_image_path': imgname,
                'normal_image_path': normal_imgname
            }

    # baseline methods
    print("Evaluate baselines")
    volume_interp_network = base_ln.get_volume_interpolation_network()
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.FINITE_DIFFERENCES
    for scale, name, imgname in EVAL_SCREEN_FD_SCALES:
        print("evaluate FD with scale", scale)
        volume_interp_network.finite_differences_stepsize = 1 / (scale * config.grid_size)
        output_stats_screen['FD%s'%name] = _eval_screen(
            base_ln, 'FD%s'%imgname, LoadedModel.EvaluationMode.TENSORCORES_MIXED)
    print("evaluate AD")
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.ADJOINT_METHOD
    output_stats_screen['AD'] = _eval_screen(
        base_ln, 'AD', LoadedModel.EvaluationMode.TENSORCORES_MIXED)
    # no-grad network
    print("evaluate no-grad network")
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.OFF_OR_DIRECT
    output_stats_screen['nograd'] = _eval_screen(
        base_ln, 'nograd', LoadedModel.EvaluationMode.TENSORCORES_MIXED)
    output_stats_screen['nograd']['compressed_size'] = lns['nograd'][2]
    output_stats_screen['nograd']['compression'] = \
        reference_volume_size / lns['nograd'][2]
    # densitygrad networks
    for i in range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX + 1):
        print("evaluate network", i)
        ln: LoadedModel = lns[i][0]
        volume_interp_network = ln.get_volume_interpolation_network()
        volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.OFF_OR_DIRECT
        output_stats_screen['network%+02d' % i] = _eval_screen(
            ln, 'network%+02d'%i, LoadedModel.EvaluationMode.TENSORCORES_MIXED)
        output_stats_screen['network%+02d' % i]['compressed_size'] = lns[i][2]
        output_stats_screen['network%+02d' % i]['compression'] = \
            reference_volume_size / lns[i][2]
    output_stats_screen['reference'] = {'time_seconds': screen_time_reference}
    output_stats['screen'] = output_stats_screen
    torch.cuda.empty_cache()

    # EVALUATE WORLD

    print("-- EVALUATE WORLD SPACE --")
    # create dataset
    dataset = []
    volume_interpolation = base_ln.get_image_evaluator().volume
    ray_evaluator = base_ln.get_image_evaluator().ray_evaluator
    min_density = ray_evaluator.min_density
    max_density = ray_evaluator.max_density
    tf_evaluator = ray_evaluator.tf
    sampler = PlasticSampler(3)
    for i in tqdm.trange(num_batches):
        indices = np.arange(i*batch_size, (i+1)*batch_size, dtype=np.int32)
        locations = sampler.sample(indices).astype(np.float32)
        locations_gpu = torch.from_numpy(locations).to(device=device)
        densities, gradients = volume_interpolation.evaluate_with_gradients(locations_gpu)
        colors = tf_evaluator.evaluate(
            densities, min_density, max_density, gradients=gradients)
        opacities = colors[:, 3:4]
        dataset.append((
            locations,
            torch.clamp(densities, 0.0, 1.0).cpu().numpy(),
            gradients.cpu().numpy(),
            opacities.cpu().numpy()))
    dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False)
    tf_index = torch.full((batch_size,), 0, dtype=torch.int32, device=device)
    time_index = torch.full((batch_size,), 0, dtype=torch.float32, device=device)
    ensemble_index = torch.full((batch_size,), 0, dtype=torch.float32, device=device)
    network_args = [tf_index, time_index, ensemble_index, 'screen']

    output_stats_world = {}
    # no-gradient for performance
    print("No gradients for performance")
    output_stats_world['Forward-PyTorch32'] = _eval_world(
        base_ln.get_network_pytorch()[0], dataloader, network_args, no_gradients=True)
    output_stats_world['Forward-PyTorch16'] = _eval_world(
        base_ln.get_network_pytorch()[1], dataloader, network_args, no_gradients=True, input16bit=True)
    volume_interp_network = base_ln.get_volume_interpolation_network()
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.OFF_OR_DIRECT
    output_stats_world['Forward-TensorCores-NoSaving'] = _eval_world(
        volume_interp_network, dataloader, no_gradients=True)
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.ADJOINT_METHOD
    output_stats_world['Forward-TensorCores-WithSaving'] = _eval_world(
        volume_interp_network, dataloader, no_gradients=True)
    # baseline methods
    print("Evaluate baselines")
    output_stats_world['FD-PyTorch'] = _eval_world(
        network_fd, dataloader, network_args)
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.FINITE_DIFFERENCES
    for scale, name, _ in EVAL_WORLD_FD_SCALES:
        volume_interp_network.finite_differences_stepsize = 1 / (scale * config.grid_size)
        output_stats_world['FD-TensorCores%s'%name] = _eval_world(
            volume_interp_network, dataloader)
    output_stats_world['AD-PyTorch'] = _eval_world(
        network_ad, dataloader, network_args)
    volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.ADJOINT_METHOD
    for scale, name in EVAL_WORLD_AD_SCALES:
        volume_interp_network.adjoint_latent_grid_central_differences_stepsize_scale = scale
        output_stats_world['AD-TensorCores%s'%name] = _eval_world(
            volume_interp_network, dataloader)
    # densitygrad networks
    for i in range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX + 1):
        print("evaluate network", i)
        ln: LoadedModel = lns[i][0]
        volume_interp_network = ln.get_volume_interpolation_network()
        volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.OFF_OR_DIRECT
        output_stats_world['network%+02d'%i] = _eval_world(
            volume_interp_network, dataloader)
    output_stats['world'] = output_stats_world
    torch.cuda.empty_cache()

    # save statistics
    print("\n===================================== Done, save statistics")

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            return json.JSONEncoder.default(self, obj)

    with open(statistics_file, "w") as f:
        json.dump(output_stats, f, cls=NumpyEncoder)
    return statistics_file


def make_plots(cfgs: List[Tuple[Config, str]]):
    #load stats
    cfgs2 = []
    for row, (cfg, statfile) in enumerate(cfgs):
        with open(statfile, "r") as f:
            stats = json.load(f)
            cfgs2.append((cfg, stats))

    #_make_adjoint_table(cfgs2)
    #_make_fd_table(cfgs2)

    _make_performance_table(cfgs2)
    _make_synthetic_error_plots(cfgs2)
    _make_big_error_table(cfgs2)
    _make_teaser(cfgs2)

def _make_adjoint_table(cfgs: List[Tuple[Config, dict]]):
    """
    Table to analyze the stepsize for the latent grid derivative in world space
    """
    def format_stat(stat, key):
        s = stat['world'][key]['gradient_l1']['mean']
        return "%.3e"%s, s

    print("Write adjoint table table")
    with open(os.path.join(BASE_PATH, "AdjointTable.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}c|c%s@{}}\n"%("c"*len(EVAL_WORLD_AD_SCALES)))
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{%d}{c}{AD Grid Stepsize - Mean Gradient L1}\\\\\n"%(1+len(EVAL_WORLD_AD_SCALES)))
        f.write("Dataset & Torch & %s\\\\\n" % " & ".join([name for scale,name in EVAL_WORLD_AD_SCALES]))
        f.write("\\midrule\n")
        for cfg, stats in cfgs:
            f.write(cfg.name)
            f.write(" & ")
            f.write(format_stat(stats, 'AD-PyTorch')[0])
            best_stat_index = np.argmin([format_stat(stats, 'AD-TensorCores%s'%name)[1] for scale,name in EVAL_WORLD_AD_SCALES])
            for i,(scale,name) in enumerate(EVAL_WORLD_AD_SCALES):
                f.write(" & ")
                s,v = format_stat(stats, 'AD-TensorCores%s'%name)
                if i==best_stat_index:
                    f.write("\\textbf{"+s+"}")
                else:
                    f.write(s)
            f.write("\\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _make_fd_table(cfgs: List[Tuple[Config, dict]]):
    """
    Table to analyze the stepsize for the finite differences in screenspace
    """
    def format_stat(stat, key):
        s = stat['world'][key]['gradient_l1']['mean']
        return "%.3e"%s, s
    def format_scale(scale):
        if scale < 1:
            return "%d/R"%int(1/scale)
        elif scale==1:
            return "1/R"
        else:
            return "1/%dR"%int(scale)

    print("Write finite difference table")
    with open(os.path.join(BASE_PATH, "FiniteDifferenceTable.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}c|%s@{}}\n"%("c"*len(EVAL_WORLD_AD_SCALES)))
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{%d}{c}{FD Stepsize - Mean Gradient L1}\\\\\n"%(len(EVAL_WORLD_FD_SCALES)))
        f.write("Dataset & %s\\\\\n" % " & ".join([format_scale(scale) for scale,name,_ in EVAL_WORLD_FD_SCALES]))
        f.write("\\midrule\n")
        for cfg, stats in cfgs:
            f.write(cfg.name)
            best_stat_index = np.argmin([format_stat(stats, 'FD-TensorCores%s'%name)[1] for scale,name,_ in EVAL_WORLD_FD_SCALES])
            for i,(scale,name,_) in enumerate(EVAL_WORLD_FD_SCALES):
                f.write(" & ")
                s,v = format_stat(stats, 'FD-TensorCores%s'%name)
                if i==best_stat_index:
                    f.write("\\textbf{"+s+"}")
                else:
                    f.write(s)
            f.write("\\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

def _make_performance_table(cfgs: List[Tuple[Config, dict]]):
    """
    Table to analyze the stepsize for the latent grid derivative in world space
    """
    def format_stat(stat, key1, key2, base=None):
        s = stat[key1][key2]
        if 'total_time_seconds' in s:
            s = s['total_time_seconds']
        else:
            s = s['time_seconds']
        if base is None:
            return "$%.3f$"%s, s
        else:
            return "$%.3f$ ($\\times %.2f$)"%(s,s/base), s

    print("Write performance table")
    with open(os.path.join(BASE_PATH, "PerformanceTable.tex"), "w") as f:
        f.write("\\begin{tabular}{@{}c|ccc}\n")
        f.write("\\toprule\n")
        f.write(" & \\multicolumn{3}{c}{Time in seconds for an image of $%d^2$ pixels}\\\\\n" % (EVAL_SCREEN_SIZE))
        f.write("Dataset & Direct & FD & Adjoint\\\\\n")
        f.write("\\midrule\n")
        for cfg, stats in cfgs:
            f.write(cfg.name)
            s, base = format_stat(stats, 'screen', 'network-7')
            f.write(" & " + s)
            f.write(" & " + format_stat(stats, 'screen', 'FD*1', base)[0])
            f.write(" & " + format_stat(stats, 'screen', 'AD', base)[0])
            f.write("\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n\\\\%\n")

        f.write("\\begin{tabular}{@{}ccccc}\n")
        f.write("\\toprule\n")
        f.write("\\multicolumn{5}{c|}{Time in seconds for $2^{%d}$ points}\\\\\n"%(np.log2(EVAL_WORLD_NUM_POINTS)))
        f.write("Forward & Forward w/ saving & Direct & FD & Adjoint\\\\\n")
        f.write("\\midrule\n")
        for cfg, stats in cfgs:
            s, base = format_stat(stats, 'world', 'Forward-TensorCores-NoSaving')
            f.write(s)
            f.write(" & " + format_stat(stats, 'world', 'Forward-TensorCores-WithSaving', base)[0])
            f.write(" & " + format_stat(stats, 'world', 'network-7', base)[0])
            f.write(" & " + format_stat(stats, 'world', 'FD-TensorCores*1', base)[0])
            f.write(" & " + format_stat(stats, 'world', 'AD-TensorCores*4', base)[0])
            f.write("\\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")


def _make_synthetic_error_plots(cfgs: List[Tuple[Config, dict]]):
    print("Write small statistics for synthetic tests")
    PLOT = "boxplot" # "errorbar", "violinplot", "boxplot"
    YAXIS = "linear" # log, linear

    cfgs_filtered = list(filter(lambda x: x[0].synthetic, cfgs))
    num_classes = len(cfgs_filtered)
    cm = matplotlib.cm.get_cmap('viridis')
    class_colors = [
        cm(f) for f in np.linspace(0, 1, num_classes)
    ]

    X = XticksMajor = np.array([0, 1, 2])
    Xclass = 4
    Xall = np.concatenate([X + Xclass*i for i in range(num_classes)])
    Xlabels = ["FD", "Adjoint", "Direct"]
    XlabelsAll = np.concatenate([
        ["FD\n ", f"Adjoint\n$\\bf{{{cfg.human_name}}}$", "Direct\n "]
        for cfg,s in cfgs_filtered])

    violin_width = 0.8
    violin_alpha = 0.5
    marker_size = 8

    def errorbar(ax, x, s, color):
        y = np.array([s['median']]) if PLOT == 'boxplot' else np.array([s['mean']])
        if PLOT == "errorbar":
            yerr = np.array([s['std']])
            ax.errorbar([x], y, yerr=yerr, elinewidth=0.5*violin_width, color='black')
            ax.plot([x], y, color=color, marker='o', markersize=marker_size)
        elif PLOT == "violinplot":
            # simulate data
            frequencies = np.array(s['histogram']['frequencies'])
            bin_edges = np.array(s['histogram']['bin_edges'])
            MAX_POINTS = 10000
            current_points = np.sum(frequencies, dtype=np.int64)
            frequencies = (frequencies * (MAX_POINTS / current_points)).astype(np.int32)
            x1 = np.random.uniform(np.repeat(bin_edges[:-1], frequencies), np.repeat(bin_edges[1:], frequencies))
            # plot
            parts = ax.violinplot([x1], positions=[x], widths=violin_width,
                          showmeans=False, showmedians=False, showextrema=False)
            for pc in parts['bodies']:
                pc.set_facecolor(color)
                pc.set_edgecolor('black')
                pc.set_alpha(violin_alpha)
            # show mean
            ax.plot([x], y, color=color, marker='o', markersize=marker_size)
        elif PLOT == 'boxplot':
            bxpstats = s['bxpstats']
            ax.bxp(bxpstats, positions=[x], widths=violin_width, showfliers=False)
            #ax.boxplot([x1], positions=[x], widths=violin_width)
        # annotate
        if PLOT != "boxplot":
            ax.annotate("%.4f"%y, (x, y),
                        xytext=(0, 4),
                        textcoords='offset points',
                        ha='center', va='bottom')
    def plot(ax: plt.Axes, stat, lossX, color, offX):
        if not isinstance(lossX, (list, tuple)):
            lossX = [lossX]

        def get_loss(key):
            s = stat['world'][key]
            for l in lossX:
                s = s[l]
            return s

        s = get_loss('FD-TensorCores*1')
        errorbar(ax, offX + X[0], s, color=color)
        s = get_loss('AD-TensorCores*4')
        errorbar(ax, offX + X[1], s, color=color)
        s = get_loss('network%+d'%GRADIENT_WEIGHT_DEFAULT_VALUE)
        errorbar(ax, offX + X[2], s, color=color)

    fig, axes = plt.subplots(nrows=1, ncols=2, squeeze=True, figsize=(9, 2.5))

    for dset, (cfg, stats) in enumerate(cfgs_filtered):
        plot(axes[0], stats, 'length_l1', class_colors[dset], dset*Xclass)
    if YAXIS=='log':
        axes[0].set_yscale("symlog", linthresh=0.2)
    axes[0].set_xticks(Xall)
    axes[0].set_xticklabels(XlabelsAll)
    axes[0].set_title("Gradient Magnitude Error $\downarrow$")

    for dset, (cfg, stats) in enumerate(cfgs_filtered):
        plot(axes[1], stats, ['cosine_similarity', 0, 'data'], class_colors[dset], dset*Xclass)
    if YAXIS=='log':
        zero_threshold = 1e-2
        max_y = 1.01
        axes[1].set_ylim(0.0, max_y) #(-1.5, max_y)
        axes[1].set_yscale("functionlog", functions=[
            lambda x: np.maximum(zero_threshold, max_y - x),
            lambda y: np.where(y > zero_threshold, max_y - y, max_y - zero_threshold)
        ])
        axes[1].set_yticks(list(np.arange(10) * 0.1) + list(np.arange(10) * 0.01 + 0.9), minor=True)
        axes[1].set_yticks([0, 0.5, 0.9, 1], minor=False) #([-1, -0.5, 0, 0.5, 0.9, 1], minor=False)
        axes[1].set_yticklabels(["0", "0.5", "0.9", "1"]) #(["-1", "-0.5", "0", "0.5", "0.9", "1"])
        axes[1].set_yticklabels([], minor=True)
    else:
        axes[1].invert_yaxis()
    axes[1].set_xticks(Xall)
    axes[1].set_xticklabels(XlabelsAll)
    axes[1].set_title("Gradient Cosine Similarity $\downarrow$")

    fig.tight_layout()
    output_filename = os.path.join(BASE_PATH, 'GradientsAnalyticDatasets.pdf')
    fig.savefig(output_filename, bbox_inches='tight')
    print("Done, saved to", output_filename)

    # copy files
    OUT_PATH = os.path.join(BASE_PATH, "images-out")
    os.makedirs(OUT_PATH, exist_ok=True)
    IMAGE_KEYS = [
        "reference", "FD_x1", "AD",
        "network%+d"%GRADIENT_WEIGHT_DEFAULT_VALUE
    ]
    IMAGE_NAMES = [
        "Ref.", "FD", "Adjoint", "Direct"
    ]
    STAT_KEYS = [
        None,
        'FD*1',
        'AD',
        'network%+d' % GRADIENT_WEIGHT_DEFAULT_VALUE
    ]
    for cfg, stats in cfgs_filtered:
        for k in IMAGE_KEYS:
            filename = "%s-color-%s.png"%(cfg.name, k)
            in_path = os.path.join(BASE_PATH, "images", filename)
            out_path = os.path.join(OUT_PATH, filename)
            shutil.copy2(in_path, out_path)

    # make table
    LATEX_IMAGE_PREFIX = "figures/analytic/" #"images-out/"
    LATEX_IMAGE_SIZE = "%.3f\\linewidth"%(0.9/len(IMAGE_KEYS))
    with open(os.path.join(BASE_PATH, "GradientsAnalyticDatasetsImages.tex"), "w") as f:
        f.write("""
        \\setlength{\\tabcolsep}{2pt}%
        \\renewcommand{\\arraystretch}{0.4}%
        """)
        f.write("\\begin{tabular}{%s}%%\n" % ("rl" * len(IMAGE_KEYS)))
        for row, (cfg, stats) in enumerate(cfgs_filtered):
            if row>0: f.write("\\\\%\n")
            # Images
            for col, k in enumerate(IMAGE_KEYS):
                filename = "%s-color-%s_lens.png" % (cfg.name, k)
                if col>0: f.write("&%\n")
                f.write("\\multicolumn{2}{c}{\\includegraphics[width=%s]{%s}}" % (
                    LATEX_IMAGE_SIZE, LATEX_IMAGE_PREFIX+filename))
            # stats
            for i, (stat, fmt) in enumerate([
                ('ssim-color', "SSIM: %.3f"),
                ('lpips-color', "LPIPS: %.3f")]):
                f.write("\\\\%\n")
                for col, (k,n,sk) in enumerate(zip(IMAGE_KEYS, IMAGE_NAMES,STAT_KEYS)):
                    if col > 0: f.write("&%\n")
                    if i==0:
                        f.write("\multirow{2}{*}{%s}%%\n"%n)
                    if sk is None:
                        f.write("&~%\n")
                    else:
                        f.write("&{\\tiny " + (fmt % stats['screen'][sk][stat]) + "}%\n")
        f.write("\\end{tabular}%\n")
    print("Latex file written")

def _make_big_error_table(cfgs: List[Tuple[Config, dict]]):
    print("Write big error table")

    plot_type = 'violin' # 'errorbar', 'plot', 'violin'
    scale_x = 'linear' #'only_one' # 'linear' or 'like_weights'

    if scale_x == 'only_one':
        weight_indices = [GRADIENT_WEIGHT_DEFAULT_VALUE]
    else:
        weight_indices = list(range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX+1))
    weight_values = [_gradient_weight(i) for i in weight_indices]

    if scale_x == 'like_weights':
        XoffWeights = 0.2
        X = np.array([0, 0.1] + [XoffWeights + w for w in weight_values])
        XticksMajor = [0, 0.1] + [XoffWeights + w for w in weight_values[::5]]
        XticksMinor = [XoffWeights + w for w in weight_values]
        Xlabels = ["FD", "AD"] + ["%.2f"%w for w in weight_values[::5]]
    elif scale_x == 'linear':
        #assert GRADIENT_WEIGHT_RANGE_MAX == -4, "GRADIENT_WEIGHT_RANGE_MAX changed, also change plot x indexing"
        #assert GRADIENT_WEIGHT_RANGE_MIN == -8, "GRADIENT_WEIGHT_RANGE_MIN changed, also change plot x indexing"
        range_weight_values = list(range(len(weight_values)))
        X = np.array([0, 1.5] + [3 + i for i in range_weight_values])
        XticksMajor = X
        XticksMinor = X
        Xlabels = ["FD", "AD"] + ["%.4f" % w for w in weight_values]
    else: # only one example for gradient weights
        assert len(weight_indices)==1
        X = XticksMajor = np.array([0, 1, 2])
        XticksMinor = []
        Xlabels = ["FD", "AD", "ours"]


    violin_width = 0.6
    violin_alpha = 0.4
    marker_size = 8

    def errorbar(ax, x, sx, color, clip=False, plot_type=plot_type):
        y = np.array([s['mean'] for s in sx])
        yerr = np.array([s['std'] for s in sx])

        if plot_type == 'violin':
            for i, s in enumerate(sx):
                # simulate data
                frequencies = np.array(s['histogram']['frequencies'])
                bin_edges = np.array(s['histogram']['bin_edges'])
                MAX_POINTS = 10000
                current_points = np.sum(frequencies, dtype=np.int64)
                frequencies = (frequencies * (MAX_POINTS / current_points)).astype(np.int32)
                x1 = np.random.uniform(np.repeat(bin_edges[:-1], frequencies), np.repeat(bin_edges[1:], frequencies))
                # plot
                parts = ax.violinplot([x1], positions=x[i:i+1], widths=violin_width,
                              showmeans=False, showmedians=False, showextrema=False)
                for pc in parts['bodies']:
                    pc.set_facecolor(color)
                    pc.set_edgecolor('black')
                    pc.set_alpha(violin_alpha)
            # show mean
            ax.plot(x, y, color=color, marker='o', markersize=marker_size)
        elif plot_type == 'errorbar':
            if clip:
                # clip error to avoid negative numbers
                yerr2 = np.copy(yerr)
                yerr2[yerr >= y] = y[yerr >= y] * .999999
                yerr = yerr2
            ax.errorbar(x, y, yerr=yerr, color=color, marker='o', markersize=marker_size)
        elif plot_type == 'plot':
            ax.plot(x, y, color=color, marker='o', markersize=marker_size)
        else:
            raise ValueError("Unknown plot type: " + plot_type)

    #fig, axes = plt.subplots(len(cfgs), 7, squeeze=False, sharey='col', figsize=(7*5, 4*len(cfgs)))
    fig, axes = plt.subplots(len(cfgs), 7, squeeze=False, figsize=(7 * 5, 4 * len(cfgs)))
    for row, (cfg, stats) in enumerate(cfgs):
        ax0 = axes[row, 0]
        ax5 = axes[row, 1]
        ax6 = axes[row, 2]
        ax1 = axes[row, 3]
        ax2 = axes[row, 4] # ax2 = ax1.twinx()
        ax3 = axes[row, 5] # ax3 = ax1.twinx()
        ax4 = axes[row, 6]

        ax0.set_ylabel(cfg.name, fontsize='xx-large')
        if row==0:
            ax0.set_title("Reference Rendering")
            ax5.set_title("Adjoint Method")
            ax6.set_title("Best Direct Prediction")
            ax1.set_title("Density L1 $\downarrow$")
            ax2.set_title("Gradient Length L1 $\downarrow$")
            ax3.set_title("Gradient Cosine Similarity $\downarrow$")
            ax4.set_title("Image LPIPS $\downarrow$")
        if row==len(cfgs)-1:
            ax1.set_xlabel("network gradient weight")
            ax2.set_xlabel("network gradient weight")
            ax3.set_xlabel("network gradient weight")
            ax4.set_xlabel("network gradient weight")

        img = imageio.imread(os.path.join(BASE_PATH, "images", "%s-color-reference.png"%cfg.name))
        ax0.imshow(img)
        ax0.get_xaxis().set_visible(False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        ax0.tick_params(axis='both', which='both', length=0)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax0.spines[spine].set_visible(False)

        img = imageio.imread(os.path.join(BASE_PATH, "images", "%s-color-AD.png" % cfg.name))
        ax5.imshow(img)
        ax5.get_xaxis().set_visible(False)
        plt.setp(ax5.get_yticklabels(), visible=False)
        ax5.tick_params(axis='both', which='both', length=0)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax5.spines[spine].set_visible(False)

        best_lpips_index = np.argmin([stats['screen']['network%+d'%i]['lpips-color'] for i in weight_indices])
        best_lpips_index = weight_indices[best_lpips_index]
        img = imageio.imread(os.path.join(BASE_PATH, "images", "%s-color-network%+d.png" % (cfg.name, best_lpips_index)))
        ax6.imshow(img)
        ax6.get_xaxis().set_visible(False)
        plt.setp(ax6.get_yticklabels(), visible=False)
        ax6.tick_params(axis='both', which='both', length=0)
        for spine in ['top', 'right', 'bottom', 'left']:
            ax6.spines[spine].set_visible(False)

        cm = matplotlib.cm.get_cmap('viridis')
        color1 = cm(0)
        color2 = cm(0.33)
        color3 = cm(0.66)
        color4 = cm(0.99)

        def plot(ax: plt.Axes, stat, lossX, color, offx, clip=False, plot_type=plot_type):
            if not isinstance(lossX, (list, tuple)):
                lossX = [lossX]
            def get_loss(key):
                s = stat['world'][key]
                for l in lossX:
                    s = s[l]
                return s
            s = get_loss('FD-TensorCores*1')
            errorbar(ax, [X[0]+offx], [s], color=color, clip=clip, plot_type=plot_type)
            s = get_loss('AD-TensorCores*4')
            errorbar(ax, [X[1]+offx], [s], color=color, clip=clip, plot_type=plot_type)
            sx = []
            for i in weight_indices:
                sx.append(get_loss('network%+d'%i))
            errorbar(ax, X[2:]+offx, sx, color=color, clip=clip, plot_type=plot_type)
            return color

        for ax in [ax1, ax2, ax3, ax4]:
            ax.set_xticks(XticksMajor, minor=False)
            ax.set_xticks(XticksMinor, minor=True)
            ax.set_xticklabels(Xlabels, minor=False)

        #ax1.set_ylabel("density L1")
        plot(ax1, stats, 'density_l1', color1, 0)
        ax1.yaxis.label.set_color(color1)
        ax1.set_yscale("symlog", linthresh=0.1)

        #ax2.set_ylabel("gradient length L1")
        plot(ax2, stats, 'length_l1_weighted', color2, 0)
        ax2.set_yscale("symlog", linthresh=0.2)
        ax2.yaxis.label.set_color(color2)

        #ax3.set_ylabel("gradient cosine sim. $\epsilon=%.2f$"%
        #               EVAL_LENGTH_THRESHOLDS[EVAL_LENGTH_THRESHOLDS_IDX_PLOT])
        #plot(ax3, stats, ['cosine_similarity', EVAL_LENGTH_THRESHOLDS_IDX_PLOT, 'data'], color3, 0)
        plot(ax3, stats, ['cosine_similarity_weighted', 0, 'data'], color3, 0)
        #ax3.invert_yaxis()
        ax3.yaxis.label.set_color(color3)
        #ax3.spines['right'].set_position(('outward', 60))
        zero_threshold = 1e-2
        max_y = 1.01
        ax3.set_ylim(-1.5, max_y)
        ax3.set_yscale("functionlog", functions=[
            lambda x: np.maximum(zero_threshold, max_y-x),
            lambda y: np.where(y>zero_threshold, max_y-y, max_y-zero_threshold)
            ])
        ax3.set_yticks(list(np.arange(10)*0.1) + list(np.arange(10)*0.01+0.9), minor=True)
        ax3.set_yticks([-1, -0.5, 0, 0.5, 0.9, 1], minor=False)
        ax3.set_yticklabels(["-1", "-0.5", "0", "0.5", "0.9", "1"])
        ax3.set_yticklabels([], minor=True)

        ax4.plot([X[0]], [stats['screen']['FD*1']['lpips-color']], color=color4, marker='o', markersize=marker_size)
        ax4.plot([X[1]], [stats['screen']['AD']['lpips-color']], color=color4, marker='o', markersize=marker_size)
        y4 = [stats['screen']['network%+d'%i]['lpips-color'] for i in weight_indices]
        ax4.plot(X[2:], y4, color=color4, marker='o', markersize=marker_size)
        ax4.set_yscale('log')

    fig.tight_layout()
    output_filename = os.path.join(BASE_PATH, 'GradientNetworks.pdf')
    fig.savefig(output_filename, bbox_inches='tight')
    #plt.show()
    print("Done, saved to", output_filename)


def _make_teaser(cfgs: List[Tuple[Config, dict]]):
    print("Write teaser")
    IMAGE_FOLDER = os.path.join(BASE_PATH, "Teaser")
    LATEX_IMAGE_SIZE = "height=3.5cm"
    HEATMAP_SIZE = "width=7cm"
    COLUMNS = 2 # number of columns of datasets

    # filter for teaser datasets
    cfgs_filtered = list(filter(lambda x: x[0].use_in_teaser, cfgs))
    num_dsets = len(cfgs_filtered)
    assert num_dsets%COLUMNS==0

    # find best network for each dataset based on LPIPS score
    best_index = [None] * num_dsets
    best_index_raw = [None] * num_dsets
    weight_indices = list(range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX + 1))
    weight_indices_names = ["$w$="+str(w) for w in weight_indices]
    for i in range(num_dsets):
        cfg, stats = cfgs_filtered[i]
        best_lpips_index = np.argmin([stats['screen']['network%+d'%i]['lpips-color'] for i in weight_indices])
        best_index[i] = weight_indices[best_lpips_index]
        best_index_raw[i] = best_lpips_index
        print(f"Dataset {cfg.name}, best weight index: {best_index[i]}, default: {GRADIENT_WEIGHT_DEFAULT_VALUE}")

    # write LaTeX and Images
    os.makedirs(IMAGE_FOLDER, exist_ok=True)
    with open(os.path.join(IMAGE_FOLDER, "GradientTeaser-v1.tex"), "w") as f:
        f.write("""
    \\documentclass[10pt,a4paper]{standalone}
    \\usepackage{graphicx}
    \\usepackage{multirow}
    \\begin{document}

    \\newcommand{\\timesize}{0.2}%
    \\setlength{\\tabcolsep}{1pt}%
    \\renewcommand{\\arraystretch}{0.4}%
    """)

        f.write("\\begin{tabular}{%s}%%\n" % ("rl" * (4*COLUMNS)))

        # header
        NAMES = ["a) Reference", "b) Finite Differences", "c) Adjoint", "d) Direct"]
        NAMES = [v for i in range(COLUMNS) for v in NAMES]
        for i, n in enumerate(NAMES):
            if i > 0: f.write(" & ")
            f.write("\\multicolumn{2}{c}{%s}" % n)

        # statistic declaration
        STATS = [
            # key, name, value-lambda
            ('time_seconds', 'Rendering:',
             lambda v: ("%.3fs" % v) if v < 40 else ("%dm %02ds" % (int(v / 60), int(v) % 60))),
            ('ssim-color', 'SSIM {\\tiny $\\uparrow$}:', lambda v: "%.3f" % v),
            ('lpips-color', 'LPIPS {\\tiny $\\downarrow$}:', lambda v: "%.3f" % v)
        ]

        # each dataset gets its own row
        for row in range(num_dsets//COLUMNS):
            name_cols = [
                cfgs_filtered[r][0].name for r in range(row*COLUMNS, (row+1)*COLUMNS)
            ]
            stats_cols = [
                cfgs_filtered[r][1] for r in range(row*COLUMNS, (row+1)*COLUMNS)
            ]
            best_lpips_index_cols = [
                best_index[r] for r in range(row*COLUMNS, (row+1)*COLUMNS)
            ]
            f.write("\\\\%\n")

            # image + stat names
            IMAGE_NAMES_cols = [[
                "%s-color-reference" % name_cols[c],
                "%s-color-FD_x1" % name_cols[c],
                "%s-color-AD" % name_cols[c],
                "%s-color-network%+d" % (name_cols[c], best_lpips_index_cols[c]),
                # extra names, needed later for the detailed stats
                "%s-color-network%+d" % (name_cols[c], GRADIENT_WEIGHT_DEFAULT_VALUE),
                "%s-normal-reference" % name_cols[c],
                "%s-normal-network%+d" % (name_cols[c], best_lpips_index_cols[c]),
                "%s-normal-network%+d" % (name_cols[c], GRADIENT_WEIGHT_DEFAULT_VALUE),
            ] for c in range(COLUMNS)]
            STAT_NAMES_cols = [[
                "reference",
                "FD*1",
                "AD",
                "network%+d"%best_lpips_index_cols[c]
            ] for c in range(COLUMNS)]

            # images
            for col1 in range(COLUMNS):
                for col2 in range(4):
                    shutil.copy2(os.path.join(BASE_PATH, "images", IMAGE_NAMES_cols[col1][col2]+".png"),
                                 os.path.join(IMAGE_FOLDER, IMAGE_NAMES_cols[col1][col2]+".png"))
                    img = "%s_lens.png" % IMAGE_NAMES_cols[col1][col2]
                    if not (col1==0 and col2==0):
                        f.write(" &%\n")
                    f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}}%%\n" % (LATEX_IMAGE_SIZE, img))
                # extra copy for the detailed statistics
                for col2 in range(4, len(IMAGE_NAMES_cols[col1])):
                    shutil.copy2(os.path.join(BASE_PATH, "images", IMAGE_NAMES_cols[col1][col2] + ".png"),
                                 os.path.join(IMAGE_FOLDER, IMAGE_NAMES_cols[col1][col2] + ".png"))

            # statistics
            for stat_key, stat_name, stat_value in STATS:
                f.write("\\\\%\n")
                for col1 in range(COLUMNS):
                    for col2 in range(4):
                        if not (col1 == 0 and col2 == 0):
                            f.write(" &%\n")
                        net_name = STAT_NAMES_cols[col1][col2]
                        if (net_name is not None) and (stat_key in stats_cols[col1]['screen'][net_name]):
                            v = stats_cols[col1]['screen'][net_name][stat_key]
                            f.write("{\\footnotesize %s} & {\\footnotesize %s}%%\n" % (stat_name, stat_value(v)))
                        else:
                            f.write(" & %\n")

        f.write("\\end{tabular}%\n")
        f.write("\\end{document}")

    # create heatmap images
    default_weight_index = weight_indices.index(GRADIENT_WEIGHT_DEFAULT_VALUE)
    def make_heatmap(cfg: Config, stats:dict, colornormal:str, best_index: int, humanname: str):
        values_lpips = np.array([stats['screen']['network%+d' % i]['lpips-'+colornormal] for i in weight_indices])
        values_ssim = np.array([stats['screen']['network%+d' % i]['ssim-' + colornormal] for i in weight_indices])
        cmap = "rocket_r"
        fig, axes = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(14, 1.3))
        # make heatmap - SSIM
        g = sns.heatmap(values_ssim[np.newaxis,:], ax=axes[0],
                    cmap='mako',
                    annot=True, fmt='.3f',
                    annot_kws={'fontsize': 8},
                    linewidths=1, square=True,
                    xticklabels=weight_indices_names,
                    yticklabels=[f"SSIM {humanname}:"],
                    cbar=False)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=8)
        # make heatmap - LPIPS
        g = sns.heatmap(values_lpips[np.newaxis, :], ax=axes[1],
                    cmap='rocket_r',
                    annot=True, fmt='.3f',
                    annot_kws={'fontsize': 8},
                    linewidths=1, square=True,
                    xticklabels=weight_indices_names,
                    yticklabels=[f"LPIPS {humanname}:"],
                    cbar=False)
        g.set_yticklabels(g.get_yticklabels(), rotation=0)
        g.set_xticklabels(g.get_xticklabels(), rotation=0, fontsize=8)
        # annotate
        def annotate(x, c):
            rect = patches.Rectangle((x+0.05, 0.05), 0.9, 0.9, linewidth=2, edgecolor=c, fill=False)
            rect.set_clip_on(False)
            axes[1].add_patch(rect)
        annotate(default_weight_index, 'green')
        annotate(best_index, 'red')
        # save
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.01)
        output_filename = f'Heatmap_{cfg.name}_{colornormal}.pdf'
        fig.savefig(os.path.join(IMAGE_FOLDER, output_filename), bbox_inches='tight')
        plt.close(fig)
        return output_filename

    with open(os.path.join(IMAGE_FOLDER, "GradientTeaserDetailed-v1.tex"), "w") as f:
        f.write("""
    \\documentclass[10pt,a4paper]{standalone}
    \\usepackage{graphicx}
    \\usepackage{xcolor}
    \\usepackage[export]{adjustbox}
    \\usepackage{multirow}
    \\begin{document}

    \\newcommand{\\timesize}{0.2}%
    \\setlength{\\tabcolsep}{1pt}%
    \\renewcommand{\\arraystretch}{0.4}%
    \\begin{tabular}{rcccccc}%
    """)
        for i in range(num_dsets):
            cfg, stats = cfgs_filtered[i]
            if i>0: f.write("\\\\[2em]%\n")
            # name of the dataset
            f.write("\\multirow{3}{*}{\\rotatebox[origin=c]{90}{\\textbf{%s}}}%%\n"%cfg.human_name)
            # first row: heatmap
            for key,name in [("color", "Color"), ("normal", "Normal")]:
                fn = make_heatmap(cfg, stats, key, best_index_raw[i], name)
                f.write(" & \\multicolumn{3}{c}{\\includegraphics[%s]{%s}}%%\n"%(
                    HEATMAP_SIZE, fn))
            # second row: images
            f.write("\\\\%\n")
            for suffix, extra in [
                ("-color-reference", ",cfbox=black 1pt 1pt"),
                ("-color-network%+d" % GRADIENT_WEIGHT_DEFAULT_VALUE, ",cfbox=green!50!black 1pt 1pt"),
                ("-color-network%+d" % best_index[i], ",cfbox=red 1pt 1pt"),
                ("-normal-reference", ",cfbox=black 1pt 1pt"),
                ("-normal-network%+d" % GRADIENT_WEIGHT_DEFAULT_VALUE, ",cfbox=green!50!black 1pt 1pt"),
                ("-normal-network%+d" % best_index[i], ",cfbox=red 1pt 1pt")
                ]:
                f.write(" & \\includegraphics[%s%s]{%s.png}%%\n"%(
                    LATEX_IMAGE_SIZE, extra, cfg.name+suffix))
            # third row: stats
            f.write("\\\\%\n")
            wx = [GRADIENT_WEIGHT_DEFAULT_VALUE, best_index[i]]
            for key in ["color", "normal"]:
                f.write(" &\n")  # empty reference
                for j in range(2):
                    f.write(" & \\begin{tabular}{rl}")
                    network_key = "network%+d" % wx[j]
                    w = wx[j]
                    alpha = _gradient_weight(w)
                    f.write("$\\alpha$ =&$%.4f$\\\\"%alpha)
                    f.write("SSIM =&$%.3f$\\\\"%stats['screen'][network_key]['ssim-'+key])
                    f.write("LPIPS =&$%.3f$" % stats['screen'][network_key]['lpips-' + key])
                    f.write("\\end{tabular}\n")

        f.write("\\end{tabular}%\n")
        f.write("\\end{document}\n")


    print("Latex files written")

def test():
    ln = LoadedModel('volnet/results/hdf5/gradient-Sphere-w02.hdf5')

    N = 2 ** 10
    torch.manual_seed(42)
    np.random.seed(42)
    positions = torch.rand((N, 3), dtype=ln._dtype, device=ln._device)

    tf_index = torch.full((positions.shape[0],), 0, dtype=torch.int32, device=ln._device)
    time_index = torch.full((positions.shape[0],), 0, dtype=torch.float32, device=ln._device)
    ensemble_index = torch.full((positions.shape[0],), 0, dtype=torch.float32, device=ln._device)
    network_args = [tf_index, time_index, ensemble_index, 'world']

    image_evaluator = ln.get_image_evaluator()
    volume_interpolation = image_evaluator.volume

    network = ln.get_network_pytorch()[0]
    network_only_density = NetworkWrapperExtractDensity(network)
    grad_network_fd = NetworkGradientTransformer.finite_differences(network_only_density, h=1e-2)
    grad_network_ad = NetworkGradientTransformer.autodiff(network_only_density)
    grad_volume_fd = NetworkGradientTransformer.finite_differences(VolumeEvaluation(volume_interpolation), h=1e-4)

    with torch.no_grad():
        # ground truth
        densities_gt, gradients_gt = volume_interpolation.evaluate_with_gradients(positions)

        # network
        tmp = network(positions, *network_args)
        densities_network = tmp[...,:1]
        gradients_network = tmp[...,1:]
        _, gradients_fd = grad_network_fd(positions, *network_args)
        _, gradients_ad = grad_network_ad(positions, *network_args)
        densities_grid, gradients_fd_grid = grad_volume_fd(positions, *network_args)

        def density_difference(a, b):
            diff = torch.abs(a-b)
            return f"absolute difference: min={torch.min(diff).item():.4f}, " \
                   f"max={torch.max(diff).item():.4f}, mean={torch.mean(diff).item():.4f}"
        def gradient_difference(a, b):
            diff_abs = torch.abs(a-b)
            len_a = torch.linalg.norm(a, dim=1, keepdim=True)
            len_b = torch.linalg.norm(b, dim=1, keepdim=True)
            diff_length = torch.abs(len_a - len_b)
            len_a = torch.clip(len_a, min=1e-5)
            len_b = torch.clip(len_b, min=1e-5)
            N = a.shape[0]
            cosine_sim = torch.bmm((a/len_a).reshape(N, 1, 3), (b/len_b).reshape(N, 3, 1))
            return f"difference absolute: min={torch.min(diff_abs).item():.4f}, " \
                   f"max={torch.max(diff_abs).item():.4f}, mean={torch.mean(diff_abs).item():.4f}; " \
                   f"length: min={torch.min(diff_length).item():.4f}, " \
                   f"max={torch.max(diff_length).item():.4f}, mean={torch.mean(diff_length).item():.4f}; " \
                   f"cosine sim.: min={torch.min(cosine_sim).item():.4f}, " \
                   f"max={torch.max(cosine_sim).item():.4f}, mean={torch.mean(cosine_sim).item():.4f}"

        print()
        print("densities GT<->Network: ", density_difference(densities_gt, densities_network))
        print("gradients GT<->Network: ", gradient_difference(gradients_gt, gradients_network))
        print("gradients GT<->FD:      ", gradient_difference(gradients_gt, gradients_fd))
        print("gradients GT<->AutoGrad:", gradient_difference(gradients_gt, gradients_ad))
        print("densities GT<->Grid:    ", density_difference(densities_gt, densities_grid))
        print("gradients GT<->FD-Grid: ", gradient_difference(gradients_gt, gradients_fd_grid))

        ad_diff = torch.abs(gradients_gt-gradients_ad)
        max_error_pos = torch.argmax(ad_diff).item()//3
        print()
        print("Max error at index", max_error_pos)
        print(" Position:", positions[max_error_pos].cpu().numpy())
        print(" Density GT:", densities_gt[max_error_pos].cpu().numpy())
        print(" Density Network:", densities_network[max_error_pos].cpu().numpy())
        print(" Gradient GT:", gradients_gt[max_error_pos].cpu().numpy())
        print(" Gradient Network:", gradients_network[max_error_pos].cpu().numpy())
        print(" Gradient FD:", gradients_fd[max_error_pos].cpu().numpy())
        print(" Gradient AD:", gradients_ad[max_error_pos].cpu().numpy())
        #_ = grad_network_fd(positions[max_error_pos:max_error_pos+1,:], *network_args)

        # Render images
        ref_camera = ln.get_default_camera()
        ref = ln.render_reference(ref_camera, 512, 512)
        imageio.imwrite('test-reference.png', LoadedModel.convert_image(ref))

        stepsize = 0.002
        img_network = ln.render_network(ref_camera, 512, 512, LoadedModel.EvaluationMode.PYTORCH32, stepsize)
        imageio.imwrite('test-network.png', LoadedModel.convert_image(img_network))

        img_grid = ln.render_network(ref_camera, 512, 512, LoadedModel.EvaluationMode.PYTORCH32, stepsize,
                                     override_network=VolumeEvaluationWithGradient(volume_interpolation))
        imageio.imwrite('test-grid.png', LoadedModel.convert_image(img_grid))

        print("Done")

if __name__ == '__main__':
    main()
    #test()