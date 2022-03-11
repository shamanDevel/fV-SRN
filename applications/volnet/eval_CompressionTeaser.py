"""
Comparison of four images:
 a) Reference
 b) Low-res grid
 c) Only Network (Lu et al)
 d) Hybrid latent grid (ours)
All with the same compression rate.
Per Case, evaluate rendering time, SSIM, LPIPS
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import sys
import os
import shutil
import subprocess
import itertools
import imageio
import json
import torch
import io
import tqdm
from collections import OrderedDict
from typing import Callable, NamedTuple, Tuple, List, Optional
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict
import enum

BASE_PATH = 'volnet/results/eval_CompressionTeaser'
IMAGE_FOLDER = os.path.join(BASE_PATH, "images_latex")
LATEX_IMAGE_PREFIX = ""
LATEX_IMAGE_SIZE = "height=3cm"

class Config(NamedTuple):
    name: str
    settings: str
    base_resolution: int
    human_name: str
    overwrite_layers: Optional[int] = None
    overwrite_samples: Optional[str] = None
    num_refinement: int = 0
    overwrite_epochs: Optional[int] = None
    overwrite_checkpoints: Optional[int] = None
    overwrite_rebuild: Optional[int] = None

configX = [
    Config(
        name = "rm60",
        settings = "config-files/RichtmyerMeshkov-t60-v1-dvr.json",
        base_resolution = 256,
        human_name="Richtmyer-Meshkov, T=60",
    ),
    Config(
        name = "skull",
        settings = "config-files/skull-v6-dvr.json", #"neuraltextures/config-files/skull-v5-dvr.json",
        base_resolution= 256,
        human_name="Skull",
    ),
    Config(
        name="ejecta1024",
        settings="config-files/ejecta1024-v6-dvr.json",
        base_resolution=1024,
        human_name="Ejecta $1024^3",
        overwrite_samples="1024**3",
        overwrite_epochs=40,  # otherwise, it takes too long
        overwrite_checkpoints=5,
        overwrite_rebuild=21
    ),
    Config(
        name = "jet",
        settings = "config-files/LuBerger-Jet-v2-mc.json",
        base_resolution= 512,
        human_name="Jet",
        overwrite_samples= "512**3",
        num_refinement=31, #-> 32*8 = 256 samples per pixel
        overwrite_epochs=100, # otherwise, it takes too long
    )
    ]

RESULT_ONLYGRID = "OnlyGrid"
RESULT_ONLYNETWORK = "OnlyNetwork"
RESULT_HYBRID = "Hybrid"
RESULT_REFERENCE = "reference"

BEST_GRID_RESOLUTION = 32
BEST_GRID_CHANNELS = 16
BEST_NETWORK_LAYERS = 4
BEST_NETWORK_CHANNELS = 32
BEST_ACTIVATION = "SnakeAlt:1"
BEST_FOURIER_STD = -1 # NERF
BEST_FOURIER_COUNT = 14 # to fit within 32 channels

NUM_SAMPLES = "256**3"
NUM_EPOCHS = 200
STEPSIZE_PER_VOXEL = 1

def getOnlyGridResolution(latent_grid_resolution, latent_grid_channels):
    """
    Returns the resolution of the "only-grid" option for the specific latent grid config
    :param latent_grid_resolution:
    :param latent_grid_channels:
    :return:
    """
    return int(np.ceil(np.cbrt(latent_grid_resolution**3 * latent_grid_channels)))

def getNetworkParameters(channels:int, layers:int):
    """
    Returns the command line arguments for that specific network
    :param channels:
    :param layers:
    :return:
    """

    # -1 because last layer is implicit
    # The parameter lists the feature size of the hidden features
    return ["--layers", ':'.join([str(channels)] * (layers - 1))]

def getNetworkParameterCount(channels_latent, num_fourier, channels_hidden, num_hidden, channels_last):
    """
    Computes the parameter count of the network with the given channels and layers.
    See also \ref getNetworkParameters(channels_hidden, num_hidden)

    :param channels_latent:
    :param num_fourier:
    :param channels_hidden:
    :param num_hidden:
    :param channels_last:
    :return:
    """
    count = 0
    count += 3 * num_fourier # fourier matrix
    channels_first = channels_latent + 4 + 2*num_fourier
    count += channels_first * channels_hidden + channels_hidden # first layer
    # -1 because the first and last layer are already added above and below
    count += (num_hidden-2) * (channels_hidden * (channels_hidden+1))
    count += channels_hidden * channels_last + channels_last # last layer
    return count

def findNetworkDimension(target_num_parameters, channels_last):
    """
    Finds the number of channels+layers for the ResidualSine-architecture
    by Lu et al 2021 "Compressive Neural Representations of Volumetric Scalar Fields"
    """
    # Based on the correspondence with Matthew Berger, always 8 residual blocks were used
    NUM_RESIDUAL_BLOCKS = 8
    def getLayerStr(num_channels:int):
        """Returns the layer specification string for the InnerNetwork parametrization"""
        #+1 because the first layer is a Sine from the input dimension
        return ":".join(["%d"%num_channels]*(NUM_RESIDUAL_BLOCKS+1))

    # Lu et al don't use fourier features or other parametrization
    # Hence, all parameters are in the InnerNetwork
    # For simplicity, binary search until the channel count is matched
    from volnet.network import InnerNetwork

    def getNumParameters(num_channels:int):
        layers = getLayerStr(num_channels)
        net = InnerNetwork(input_channels=3, output_channels=channels_last,
                           layers=layers, activation="ResidualSine",
                           latent_size=0, split_density_and_auxiliary=False)
        params = 0
        for p in net.parameters(recurse=True):
            params += p.numel()
        return params

    # Phase one: double until we exceed the target
    high_channels = 8
    high_params = getNumParameters(high_channels)
    assert high_params<target_num_parameters, "Already a tiny network is too big"
    low_params = None
    low_channels = None
    while high_params<target_num_parameters:
        low_channels = high_channels
        low_params = high_params
        high_channels = low_channels*2
        high_params = getNumParameters(high_channels)

    # Phase two: binary search
    while high_channels-low_channels>1:
        mid_channels = np.clip((low_channels+high_channels)//2, low_channels+1, high_channels-1)
        mid_params = getNumParameters(mid_channels)
        if mid_params<target_num_parameters:
            low_channels = mid_channels
            low_params = mid_params
        else:
            high_channels = mid_channels
            high_params = mid_params

    # now pick the closest match
    if (target_num_parameters-low_params) < (high_params-target_num_parameters):
        best_channels = low_channels
        best_params = low_params
    else:
        best_channels = high_channels
        best_params = high_params

    return getLayerStr(best_channels), best_params


def get_hdf5_file(config: Config, case:str):
    assert case in [RESULT_HYBRID, RESULT_ONLYGRID, RESULT_REFERENCE, RESULT_ONLYNETWORK]
    return os.path.join(BASE_PATH, "hdf5", config.name + "-" + case + ".hdf5")

def main():
    cfgs = []
    for config in configX:
        print("\n==========================================")
        print(config.name)
        print("==========================================")

        best_network_layers = config.overwrite_layers or BEST_NETWORK_LAYERS
        training_samples = config.overwrite_samples or NUM_SAMPLES

        reference_memory = config.base_resolution**3
        only_grid_resolution = getOnlyGridResolution(BEST_GRID_RESOLUTION, BEST_GRID_CHANNELS)
        hybrid_parameters = getNetworkParameterCount(
            BEST_GRID_CHANNELS, BEST_FOURIER_COUNT, BEST_NETWORK_CHANNELS, best_network_layers, 1)
        target_memory = BEST_GRID_RESOLUTION**3 * BEST_GRID_CHANNELS + 2 * hybrid_parameters # 1byte for voxel, 2byte for weights
        onlynet_layer_str, onlynet_num_parameters = findNetworkDimension(target_memory//2, 1)
        print("Reference memory:",reference_memory)
        print("only-grid resolution:", only_grid_resolution)
        print("hybrid parameters:", hybrid_parameters,"--> memory:", target_memory, "(%.3f%% of reference)"%(target_memory/reference_memory*100))
        print("only-network layers:", onlynet_layer_str, ", parameters:", onlynet_num_parameters,
              "-> memory:", 2*onlynet_num_parameters, "(%.3f%% of reference)"%((2*onlynet_num_parameters)/reference_memory*100))
        compression = target_memory/reference_memory

        train(config, onlynet_layer_str, best_network_layers, training_samples)

        stepsize = 1.0 / (STEPSIZE_PER_VOXEL * config.base_resolution)
        statistics_file = eval(config, only_grid_resolution, stepsize, compression)

        copy_images(config)
        cfgs.append((config, statistics_file, compression))
    make_table_layout1(cfgs)
    make_table_layout2(cfgs)
    print_stats(cfgs)

def train(config: Config, onlynet_layer_str:str, best_network_layers, training_samples):

    epochs = config.overwrite_epochs or NUM_EPOCHS
    save_frequency = config.overwrite_checkpoints or 20
    rebuild = config.overwrite_rebuild or 51
    common_args = [
        sys.executable, "volnet/train_volnet.py",
        config.settings,
        "--train:mode", "world",
        "--train:samples", training_samples,
        "--train:sampler_importance", "0.01",
        '--rebuild_dataset', str(rebuild),
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "-l1", "1",
        "--lr_step", "100",
        "-i", str(epochs),
        "--logdir", BASE_PATH+'/log',
        "--modeldir", BASE_PATH+'/model',
        "--hdf5dir", BASE_PATH+'/hdf5',
        '--save_frequency', str(save_frequency)
    ]

    onlynet_args = [
        "--layers", onlynet_layer_str,
        "--train:batchsize", "64*64*32",
        "--activation", "ResidualSine", #Lu et al. uses Residual-Sine blocks and no fourier features
        "--fouriercount", "0",
        '-lr', '0.00005',
        "--name", config.name + "-" + RESULT_ONLYNETWORK
    ]

    hybrid_args = getNetworkParameters(BEST_NETWORK_CHANNELS, best_network_layers) + [
        "--train:batchsize", "64*64*128",
        "--activation", BEST_ACTIVATION,
        '--fouriercount', str(BEST_FOURIER_COUNT),
        '--fourierstd', str(BEST_FOURIER_STD),
        '--volumetric_features_channels', str(BEST_GRID_CHANNELS),
        '--volumetric_features_resolution', str(BEST_GRID_RESOLUTION),
        '-lr', '0.01',
        "--name", config.name + "-" + RESULT_HYBRID
    ]

    def run(args, filename):
        if os.path.exists(filename):
            print("Skipping", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)

    print("TRAINING")
    run(common_args + onlynet_args, get_hdf5_file(config, RESULT_ONLYNETWORK))
    run(common_args + hybrid_args, get_hdf5_file(config, RESULT_HYBRID))

def eval(config: Config, only_grid_resolution:int, stepsize:float, compression:float):
    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats-%s.json'%config.name)
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file
    print(f"Stepsize: {stepsize} (1/{1/stepsize})")

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    num_cameras = 1 # we only show one image
    width = 1024
    height = 1024
    timer = pyrenderer.GPUTimer()

    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    #rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = True
    rendering_mode_no_tc = LoadedModel.EvaluationMode.PYTORCH16

    output_stats = {
        "name": config.name,
        "settings": config.settings,
        "base_resolution": config.base_resolution,
        "compression": compression
    }
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln, mode, reference_images, stepsize, filename_template=None,
                      do_ssim=False, do_lpips=False, render_ref=False, warmup=True):
        timingsX = []
        ssimX = []
        lpipsX = []
        stepsX = []
        for i in range(num_cameras):
            num_attempts = 2 if (i==0 and warmup) else 1 # warmup
            for j in range(num_attempts):
                timer2 = None if j<num_attempts-1 else timer
                if render_ref:
                    current_image = ln.render_reference(
                        cameras[i], width, height,
                        stepsize_world=stepsize, timer=timer2,
                        num_refine=config.num_refinement)
                else:
                    if not ln.is_tensorcores_available():
                        mode = rendering_mode_no_tc
                    current_image = ln.render_network(
                        cameras[i], width, height, mode,
                        stepsize, timer=timer2,
                        num_refine=config.num_refinement)
            timingsX.append(timer.elapsed_milliseconds()/1000.0)
            if filename_template is not None:
                imageio.imwrite(
                    filename_template % i,
                    LoadedModel.convert_image(current_image))
            if do_ssim:
                ssimX.append(ssim_loss(current_image, reference_images[i]).item())
            if do_lpips:
                lpipsX.append(lpips_loss(current_image, reference_images[i]).item())
            stepsX.append(ln.get_max_steps(cameras[i], width, height, stepsize))
        return \
            (np.mean(timingsX), np.std(timingsX)), \
            (np.mean(ssimX), np.std(ssimX)) if do_ssim else (np.NaN, np.NaN), \
            (np.mean(lpipsX), np.std(lpipsX)) if do_lpips else (np.NaN, np.NaN), \
            (np.mean(stepsX), np.std(stepsX))

    # load networks
    def load_and_save(case:str):
        filename = get_hdf5_file(config, case)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            return None, None
        try:
            ln = LoadedModel(filename, force_config_file=config.settings)
            if enable_preintegration:
                ln.enable_preintegration(True, convert_to_texture=True)
            ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
            return ln
        except Exception as e:
            print("Unable to load '%s':" % filename, e)
            return None

    reference_images = None
    image_folder = os.path.join(BASE_PATH, "images_%s"%config.name)
    os.makedirs(image_folder, exist_ok=True)

    # collect models
    lns = {
        RESULT_ONLYNETWORK: load_and_save(RESULT_ONLYNETWORK),
        RESULT_HYBRID: load_and_save(RESULT_HYBRID)
    }
    base_ln = lns[RESULT_ONLYNETWORK]

    # render reference
    image_folder_reference = os.path.join(image_folder, RESULT_REFERENCE)
    os.makedirs(image_folder_reference, exist_ok=True)
    print("\n===================================== Render reference")
    cameras = base_ln.get_rotation_cameras(num_cameras)
    reference_images = [None] * num_cameras
    timingsX = []
    for i in range(num_cameras):
        if i==0:
            _ = base_ln.render_reference(cameras[i], width, height, stepsize_world=stepsize, timer=None) # warmup
        reference_images[i] = base_ln.render_reference(cameras[i], width, height, stepsize_world=stepsize, timer=timer, num_refine=config.num_refinement)
        timingsX.append(timer.elapsed_milliseconds()/1000.0)
        imageio.imwrite(
            os.path.join(image_folder_reference, 'reference%03d.png' % i),
            LoadedModel.convert_image(reference_images[i]))
    output_stats[RESULT_REFERENCE] = {'inference_time_seconds': (np.mean(timingsX), np.std(timingsX))}

    # render networks
    for key in [RESULT_HYBRID, RESULT_ONLYNETWORK]:
        ln = lns[key]
        if ln is None:
            print("Skip", key, ", network is None")
            continue
        print("\n===================================== Render", key)
        image_folder_screen = os.path.join(image_folder, "%s" % key)
        os.makedirs(image_folder_screen, exist_ok=True)
        warmup = False if (key==RESULT_ONLYNETWORK and config.num_refinement>0) else True
        time, ssim, lpips, steps = compute_stats(
            ln, rendering_mode, reference_images, stepsize,
            os.path.join(image_folder_screen, '%s%%03d.png'%key),
            True, True, warmup=warmup)
        output_stats[key] = {
            'inference_time_seconds': time,
            'ssim': ssim,
            'lpips': lpips,
            'training_time_seconds': ln.training_time_seconds(),
            'steps:': steps
        }

    # reference for just the grid
    print("Render", RESULT_ONLYGRID, "with a resolution of", only_grid_resolution)
    base_volume_interpolation = base_ln.get_image_evaluator().volume
    base_volume = base_volume_interpolation.volume()
    # Fix for new grid resolution behavior (better match for different grid resolutions)
    # Render reference again
    old_is_new_behavior = base_volume_interpolation.grid_resolution_new_behavior
    base_volume_interpolation.grid_resolution_new_behavior = True
    for i in range(num_cameras):
        reference_images[i] = base_ln.render_reference(cameras[i], width, height)
    # create a resampled grid for that
    new_volume = base_volume.create_scaled(only_grid_resolution, only_grid_resolution, only_grid_resolution)
    # render with that
    base_volume_interpolation.setSource(new_volume, 0)
    image_folder_screen = os.path.join(image_folder, "%s" % RESULT_ONLYGRID)
    os.makedirs(image_folder_screen, exist_ok=True)
    time, ssim, lpips, steps = compute_stats(
        base_ln, rendering_mode, reference_images, stepsize,
        os.path.join(image_folder_screen, '%s%%03d.png'%RESULT_ONLYGRID),
        True, True, render_ref=True)
    output_stats[RESULT_ONLYGRID] = {
        'inference_time_seconds': time,
        'ssim': ssim,
        'lpips': lpips,
        'steps': steps
    }
    base_volume_interpolation.grid_resolution_new_behavior = old_is_new_behavior

    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file, "w") as f:
        json.dump(output_stats, f)
    return statistics_file


def _calcCropAndSave(files_in, files_out):
    PAD = 10
    images = []
    minX = 1000000
    minY = 1000000
    maxX = 0
    maxY = 0
    sizeX = 0
    sizeY = 0
    # compute
    for fi in files_in:
        img = imageio.imread(fi)
        images.append(img)
        sizeX, sizeY, _ = img.shape
        alpha = img[:, :, 3]
        mask = alpha > 0.01 * 1.0
        reduceX = np.sum(mask, axis=1)
        reduceY = np.sum(mask, axis=0)
        indicesX = np.where(reduceX>0.5)[0]
        indicesY = np.where(reduceY>0.5)[0]
        minX = min(minX, indicesX[0])
        maxX = max(maxX, indicesX[-1])
        minY = min(minY, indicesY[0])
        maxY = max(maxY, indicesY[-1])

    # pad and clamp
    minX = max(0, minX-PAD)
    minY = max(0, minY-PAD)
    maxX = min(sizeX, maxX+PAD+1)
    maxY = min(sizeY, maxY+PAD+1)
    print(f"Cropping: {minX}:{maxX}, {minY}:{maxY}")

    # crop and save
    for img, fo in zip(images, files_out):
        img = img[minX:maxX, minY:maxY, :]
        imageio.imwrite(fo, img)

    return minX, minY, maxX, maxY

def _useCropAndSave(files_in, files_out, crop):
    minX, minY, maxX, maxY = crop
    for fi, fo in zip(files_in, files_out):
        img = imageio.imread(fi)
        img = img[minX:maxX, minY:maxY, :]
        imageio.imwrite(fo, img)

def copy_images(config:Config):
    print("\n===================================== Make Plots")

    os.makedirs(IMAGE_FOLDER, exist_ok=True)

    # copy images
    crop = _calcCropAndSave(
        [os.path.join(BASE_PATH, "images_%s"%config.name, RESULT_REFERENCE, RESULT_REFERENCE+"000.png")],
        [os.path.join(IMAGE_FOLDER, "%s_%s.png" % (config.name, RESULT_REFERENCE))]
    )
    for key in [RESULT_HYBRID, RESULT_ONLYGRID, RESULT_ONLYNETWORK]:
        _useCropAndSave(
            [os.path.join(BASE_PATH, "images_%s" % config.name, key, key + "000.png")],
            [os.path.join(IMAGE_FOLDER, "%s_%s.png" % (config.name, key))],
            crop)

def make_table_layout1(cfg: List[Tuple[Config, str, float]]):

    with open(os.path.join(IMAGE_FOLDER, "CompressionTeaser-layout1.tex"), "w") as f:
        f.write("""
\\documentclass[10pt,a4paper]{standalone}
\\usepackage{graphicx}
\\usepackage{multirow}
\\begin{document}

\\newcommand{\\timesize}{0.2}%
\\setlength{\\tabcolsep}{2pt}%
\\renewcommand{\\arraystretch}{0.4}%
""")

        #load stats
        stats = dict()
        for config,statsfile,compression in cfg:
            with open(statsfile, "r") as f2:
                stats[config.name] = json.load(f2)
                stats[config.name]['COMPRESSION-RATIO'] = 1/compression

        num_configs = len(configX)
        f.write("\\begin{tabular}{%s}%%\n" % ("rl" * (2*num_configs)))

        def statsTableOld(label, stat1, stat2, stat3):
            s = f"""
\\begin{{tabular}}{{rl}}%
\multirow{{3}}{{*}}{{{label}~}}%
&{{\\small {stat1}}}\\\\&{{\\small {stat2}}}\\\\&{{\\small {stat3}}}%
\\end{{tabular}}%
"""
            return s.lstrip()

        # header
        for i, config in enumerate(configX):
            if i>0: f.write("&%\n")
            f.write("\\multicolumn{4}{c}{%s, 1:%.1f}" % (config.human_name, stats[config.name]['COMPRESSION-RATIO']))

        # first row: reference, only grid
        f.write("\\\\%\n")
        first_column = True
        for config in configX:
            image_reference = "%s%s_%s.png" % (LATEX_IMAGE_PREFIX, config.name, RESULT_REFERENCE)
            image_onlygrid = "%s%s_%s.png" % (LATEX_IMAGE_PREFIX, config.name, RESULT_ONLYGRID)
            if not first_column:
                f.write("&%\n")
            first_column = False
            f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}}&" % (LATEX_IMAGE_SIZE, image_reference))
            f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}} " % (LATEX_IMAGE_SIZE, image_onlygrid))
        for i,(stat,fmt) in enumerate([
                ('inference_time_seconds', "Rendering: %.3fs"),
                ('ssim', "SSIM: %.3f"),
                ('lpips', "LPIPS: %.3f")]):
            f.write("\\\\%\n")
            first_column = True
            for config in configX:
                if not first_column:
                    f.write("&%\n")
                first_column = False
                if i==0:
                    f.write("\multirow{3}{*}{a) Ref.~}%\n")
                if i==0:
                    f.write("&{\\tiny " + fmt%stats[config.name][RESULT_REFERENCE][stat][0] + "}&\n")
                else:
                    f.write("&~&\n")
                if i==0:
                    f.write("\multirow{3}{*}{b) Grid~}%\n")
                f.write("&{\\tiny " + fmt%stats[config.name][RESULT_ONLYGRID][stat][0] + "}\n")

        # second row: hybrid, ours
        f.write("\\\\%\n")
        first_column = True
        for config in configX:
            image_onlynet = "%s%s_%s.png" % (LATEX_IMAGE_PREFIX, config.name, RESULT_ONLYNETWORK)
            image_hybrid = "%s%s_%s.png" % (LATEX_IMAGE_PREFIX, config.name, RESULT_HYBRID)
            if not first_column:
                f.write("&%\n")
            first_column = False
            f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}}&" % (LATEX_IMAGE_SIZE, image_onlynet))
            f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}} " % (LATEX_IMAGE_SIZE, image_hybrid))
        for i, (stat, fmt) in enumerate([
            ('training_time_seconds', lambda v: "Training: %dm %02ds"%(int(v/60), v%60)),
            ('inference_time_seconds', lambda v:"Rendering: %.3fs"%v[0]),
            ('ssim', lambda v: "SSIM: %.3f"%v[0]),
            ('lpips', lambda v: "LPIPS: %.3f"%v[0])]):
            f.write("\\\\%\n")
            first_column = True
            for config in configX:
                if not first_column:
                    f.write("&%\n")
                first_column = False
                if i == 0:
                    f.write("\multirow{3}{*}{c) V-SRN~}%\n")
                f.write("&{\\tiny " + fmt(stats[config.name][RESULT_ONLYNETWORK][stat]) + "}&\n")
                if i == 0:
                    f.write("\multirow{3}{*}{d) fV-SRN~}%\n")
                f.write("&{\\tiny " + fmt(stats[config.name][RESULT_HYBRID][stat]) + "}\n")

        f.write("\\end{tabular}%\n")
        f.write("\\end{document}")

    print("Latex file written")


def make_table_layout2(cfg: List[Tuple[Config, str, float]]):
    # load stats
    stats = dict()
    for config, statsfile, compression in cfg:
        with open(statsfile, "r") as f2:
            stats[config.name] = json.load(f2)
            stats[config.name]['COMPRESSION-RATIO'] = 1 / compression

    num_configs = len(configX)
    if num_configs%2!=0:
        print("Unable to create Layout 2, len(configX) must be even")
        return

    with open(os.path.join(IMAGE_FOLDER, "CompressionTeaser-layout2.tex"), "w") as f:
        f.write("""
\\documentclass[10pt,a4paper]{standalone}
\\usepackage{graphicx}
\\usepackage{multirow}
\\begin{document}

\\newcommand{\\timesize}{0.2}%
\\setlength{\\tabcolsep}{1pt}%
\\renewcommand{\\arraystretch}{0.4}%
""")

        f.write("\\begin{tabular}{%s}%%\n" % ("rl" * (8)))

        # header
        NAMES = ["a) Reference", "b) Low-Res. Grid", "c) V-SRN (Lu~\\textit{et~al.})", "d) fV-SRN (ours)"]
        NAMES = NAMES + NAMES
        for i,n in enumerate(NAMES):
            if i>0: f.write(" & ")
            f.write("\\multicolumn{2}{c}{%s}"%n)
        f.write("\\\\%\n")

        RESULTS = [RESULT_REFERENCE, RESULT_ONLYGRID, RESULT_ONLYNETWORK, RESULT_HYBRID]
        STATS = [
            # key, name, value-lambda
            ('training_time_seconds', 'Training:', lambda v: "%dm %02ds"%(int(v/60), v%60)),
            ('inference_time_seconds', 'Rendering:', lambda v:("%.3fs"%v) if v < 40 else ("%dm %02ds"%(int(v/60), int(v)%60))),
            ('ssim', 'SSIM {\\tiny $\\uparrow$}:', lambda v: "%.3f"%v),
            ('lpips', 'LPIPS {\\tiny $\\downarrow$}:', lambda v: "%.3f"%v)
        ]

        # row by row, col by col
        for row in range(num_configs//2):
            if row>0: f.write("\\\\%\n")
            # images
            for col1 in range(2):
                config = configX[col1 + 2*row]
                for col2 in range(4):
                    img = "%s%s_%s.png" % (LATEX_IMAGE_PREFIX, config.name, RESULTS[col2])
                    if not (col1==0 and col2==0):
                        f.write(" &%\n")
                    f.write("\\multicolumn{2}{c}{\\includegraphics[%s]{%s}}%%\n" % (LATEX_IMAGE_SIZE, img))
            # stats
            for stat_key, stat_name, stat_value in STATS:
                f.write("\\\\%\n")
                for col1 in range(2):
                    config = configX[col1 + 2*row]
                    for col2 in range(4):
                        if not (col1 == 0 and col2 == 0):
                            f.write(" &%\n")
                        s = stats[config.name][RESULTS[col2]]
                        if stat_key in s:
                            v = s[stat_key]
                            try:
                                v = v[0]
                            except TypeError:
                                pass
                            f.write("{\\footnotesize %s} & {\\footnotesize %s}%%\n"%(stat_name, stat_value(v)))
                        else:
                            f.write(" & %\n")

        f.write("\\end{tabular}%\n")
        f.write("\\end{document}")

    print("Latex file written")


def print_stats(cfg: List[Tuple[Config, str, float]]):
    # load stats
    stats = dict()
    for config, statsfile, compression in cfg:
        with open(statsfile, "r") as f2:
            stats[config.name] = json.load(f2)
            stats[config.name]['COMPRESSION-RATIO'] = 1 / compression

    o = io.StringIO()
    o.write("Dataset   | Training Speedup | Inference Speedup\n")
    for config, _, _ in cfg:
        train_onlynet = stats[config.name][RESULT_ONLYNETWORK]['training_time_seconds']
        train_hybrid = stats[config.name][RESULT_HYBRID]['training_time_seconds']
        eval_onlynet = stats[config.name][RESULT_ONLYNETWORK]['inference_time_seconds'][0]
        eval_hybrid = stats[config.name][RESULT_HYBRID]['inference_time_seconds'][0]
        o.write(f"{config.human_name}  |  {train_onlynet/train_hybrid}  |  {eval_onlynet/eval_hybrid}\n")

    o = o.getvalue()
    with open(os.path.join(BASE_PATH, "Speedup.txt"), 'w') as f:
        f.write(o)
    print(o)

if __name__ == '__main__':
    main()