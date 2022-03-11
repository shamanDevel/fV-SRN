"""
Script to evaluate the network quality if it is trained for densities or for colors.
They are tested with world-space training and the best configuration from eval_network_configs
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
from typing import Callable
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

BEST_ACTIVATION = "SnakeAlt:1"
IMAGE_PATTERN = "ensemble{ensemble:04d}-time{timestep:04d}-camera{camera:04d}.png"

BASE_PATH = 'volnet/results/eval_TimeVolumetricFeatures'

configX = [
        ("plume-time", [
            "config-files/plumeEnsemble-v1-dvr.json",
            "--volume_filenames",
            "volumes/ScalarFlow/sim_{ensemble:06d}/volume_{time:06d}.cvol",
            "--time_keyframes", "30:101:10",
            "--time_train", "30:101:5",
            "--time_val", "30:101:2",
        ], 178), # <- base resolution
        ("ejecta-time", [
            "config-files/ejecta-ensemble-v6.json",
            "--volume_filenames",
            "volumes/Ejecta/snapshot_{time:03d}_256.cvol",
            "--time_keyframes", "10:91:10",
            "--time_train", "10:91:5",
            "--time_val", "10:91:2",
        ], 256),
        ("rm-time", [
            "config-files/RichtmyerMeshkov-ensemble-v3.json",
            "--volume_filenames",
            "volumes/RichtmyerMeshkov/ppm-t{time:04d}.cvol",
            "--time_keyframes", "10:91:10",
            "--time_train", "10:91:5",
            "--time_val", "10:91:2",
        ], 256),
    ]

def getRange(cfg, tag):
    idx = cfg[1].index(tag)
    range_str = cfg[1][idx + 1]
    range_split = range_str.split(':')
    range_int = map(int, range_split)
    return range(*range_int)

networkX = []
networkChannelsX = [32]
networkLayersX = [4]
def networkFilename(channels, layers):
    return "l%dx%d"%(channels, layers)
for channels in networkChannelsX:
    for layers in networkLayersX:
        parameters = (channels * (channels+1)) * layers
        networkX.append((networkFilename(channels, layers), channels, layers, parameters))
networkX.sort(key=lambda x: x[3])

fourierX = [("fNeRF14", (-1, 14))]

volumetricFeaturesX = [
    ("G%dC%d"%(g,c), [
        '--volumetric_features_channels', str(c),
        '--volumetric_features_resolution', str(g),
        '--volumetric_features_time_dependent',
        '--time_features', str(c)],
     (g, c))
    for (g,c) in itertools.product([32], [16])
]

timeX = [
    ("none", []),
    ("direct", ['--use_time_direct']),
    ("fourier", ['--num_time_fourier', '4']),
    ("both", ['--use_time_direct', '--num_time_fourier', '4'])
]

def main():
    configs = collect_configurations()
    train(configs)
    statistics_file = eval(configs)
    #test_video()
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    for config, network, fourier, volumetricFeatures, time in itertools.product(
            configX, networkX, fourierX, volumetricFeaturesX, timeX):
        filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
            config[0], network[0], fourier[0], volumetricFeatures[0], time[0])
        cfgs.append((config[1], network[1:], fourier[1], volumetricFeatures[1], time[1], filename))

    # special: time-fixed grid
    for config, network, fourier, volumetricFeatures in itertools.product(configX, networkX, fourierX, volumetricFeaturesX):
        g, c = volumetricFeatures[2]
        volFeaturesArgs = [
            '--volumetric_features_channels', str(c),
            '--volumetric_features_resolution', str(g)]
        timeArgs = ['--use_time_direct']
        filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
            config[0], network[0], fourier[0], volumetricFeatures[0]+"-steady", 'direct')
        cfgs.append((config[1], network[1:], fourier[1], volFeaturesArgs, timeArgs, filename))

    # special: Lu&Berger network
    for network, fourier, volumetricFeatures in itertools.product(networkX, fourierX, volumetricFeaturesX):
        g, c = volumetricFeatures[2]
        f = fourier[1][1]
        from volnet.eval_CompressionTeaser import getNetworkParameterCount, findNetworkDimension
        hybrid_parameters = getNetworkParameterCount(
            c, f, network[1], network[2], 1)
        target_memory = g ** 3 * c + 2 * hybrid_parameters  # 1byte for voxel, 2byte for weights
        onlynet_layer_str, onlynet_num_parameters = findNetworkDimension(target_memory // 2, 1)
        onlynet_args = [
            "--layers", onlynet_layer_str,
            "--train:batchsize", "64*64*32",
            "--activation", "ResidualSine",  # Lu et al. uses Residual-Sine blocks and no fourier features
            "--fouriercount", "0",
            '-lr', '0.00005',
            '--use_time_direct'
        ]
        for config in configX:
            filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
                config[0], network[0], fourier[0], volumetricFeatures[0] + "-LuBerger", 'LuBerger')
            cfgs.append((config[1], network[1:], fourier[1], onlynet_args, [], filename))

    return cfgs

def get_args_and_hdf5_file(cfg):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :return: args, filename
    """

    common_parameters = [
        "--train:mode", "world",
        "--train:samples", "256**3",
        "--train:batchsize", "64*64*128",
        "--train:sampler_importance", "0.01",
        '--rebuild_dataset', '51',
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "--activation", BEST_ACTIVATION,
        "-l1", "1",
        "--lr_step", "100",
        "-i", "500",
        "--logdir", 'volnet/results/eval_TimeVolumetricFeatures/log',
        "--modeldir", 'volnet/results/eval_TimeVolumetricFeatures/model',
        "--hdf5dir", 'volnet/results/eval_TimeVolumetricFeatures/hdf5',
        '--save_frequency', '50'
    ]

    def getNetworkParameters(network):
        channels, layers, params = network
        return ["--layers", ':'.join([str(channels)] * (layers - 1))]

    def getFourierParameters(network, fourier):
        channels, layers, params = network
        std, count = fourier
        return ['--fouriercount', str(count), '--fourierstd', str(std)]

    config, network, fourier, volumetric, time, filename = cfg

    launcher = [sys.executable, "volnet/train_volnet.py"]
    args = launcher + config + \
           common_parameters + \
           getNetworkParameters(network) + \
           getFourierParameters(network, fourier) + \
           volumetric + time + ['--name', filename]

    hdf5_file = os.path.join(BASE_PATH, 'hdf5', filename + ".hdf5")
    return args, hdf5_file, filename

def train(configs):
    print("Configurations:", len(configs))
    for cfg in configs:
        args, filename, outputname = get_args_and_hdf5_file(cfg)
        if os.path.exists(filename):
            print("Skipping test", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval(configs):
    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats.json')
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    num_cameras = 4# 64 # 64 just takes forever
    width = 512
    height = 512
    STEPSIZE = 1/512
    timer = pyrenderer.GPUTimer()

    LIMIT_NUM_TIMESTEPS = None # 11

    if os.name == 'nt':  # windows
        rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    else:
        rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = True
    rendering_mode_no_tc = LoadedModel.EvaluationMode.PYTORCH16

    output_stats = []
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def render_reference_images(ln:LoadedModel, filename_template=None):
        assert ln.is_time_dependent()
        num_timesteps = ln.max_timestep() - ln.min_timestep() + 1
        num_ensembles = ln.max_ensemble() - ln.min_ensemble() + 1
        timingsX = np.empty((num_ensembles, num_timesteps, num_cameras))
        print("Memory for reference images:", utils.humanbytes(
            num_ensembles * num_timesteps * num_cameras * 4 * height * width * 4
        ))
        imagesX = np.zeros((num_ensembles, num_timesteps, num_cameras, 4, height, width))
        for ensemble in range(ln.min_ensemble(), ln.max_ensemble()+1):
            for timestep in range(ln.min_timestep(), ln.max_timestep()+1):
                for cam in range(num_cameras):
                    ensemble_array_idx = ensemble - ln.min_ensemble()
                    timestep_array_idx = timestep - ln.min_timestep()
                    if LIMIT_NUM_TIMESTEPS is not None and timestep_array_idx>LIMIT_NUM_TIMESTEPS:
                        continue
                    print("t=", timestep)
                    current_image = ln.render_reference(
                        cameras[cam], width, height,
                        timestep=timestep, ensemble=ensemble,
                        timer=timer)
                    timingsX[ensemble_array_idx, timestep_array_idx, cam] = timer.elapsed_milliseconds()
                    if filename_template is not None:
                        imageio.imwrite(
                            filename_template.format(ensemble=ensemble, timestep=timestep, camera=cam),
                            LoadedModel.convert_image(current_image))
                    imagesX[ensemble_array_idx, timestep_array_idx, cam, ...] = current_image[0].detach().cpu().numpy()
        return imagesX

    def compute_stats(ln:LoadedModel, mode, reference_images, stepsize, filename_template=None,
                      do_ssim=False, do_lpips=False,
                      eval_f: Callable[[int, int, int, pyrenderer.GPUTimer], torch.Tensor]=None):
        assert ln.is_time_dependent()
        num_timesteps = ln.max_timestep() - ln.min_timestep() + 1
        num_ensembles = ln.max_ensemble() - ln.min_ensemble() + 1
        timingsX = np.zeros((num_ensembles, num_timesteps, num_cameras))
        ssimX = np.zeros((num_ensembles, num_timesteps, num_cameras))
        lpipsX = np.zeros((num_ensembles, num_timesteps, num_cameras))
        with tqdm.tqdm(total=num_ensembles*num_timesteps*num_cameras) as pbar:
            for ensemble in range(ln.min_ensemble(), ln.max_ensemble()+1):
                for timestep in range(ln.min_timestep(), ln.max_timestep()+1):
                    for cam in range(num_cameras):
                        ensemble_array_idx = ensemble - ln.min_ensemble()
                        timestep_array_idx = timestep - ln.min_timestep()
                        if LIMIT_NUM_TIMESTEPS is not None and timestep_array_idx > LIMIT_NUM_TIMESTEPS:
                            continue
                        if eval_f is None:
                            current_image = ln.render_network(
                                cameras[cam], width, height,
                                mode if ln.is_tensorcores_available() else rendering_mode_no_tc,
                                stepsize,
                                timestep = timestep, ensemble=ensemble,
                                timer=timer)
                        else:
                            current_image = eval_f(ensemble, timestep, cam, timer)
                        reference_image = torch.from_numpy(
                            reference_images[ensemble_array_idx, timestep_array_idx, cam, ...])
                        reference_image = reference_image.unsqueeze(0).to(
                            dtype=current_image.dtype, device=current_image.device)
                        timingsX[ensemble_array_idx, timestep_array_idx, cam] = timer.elapsed_milliseconds()
                        if filename_template is not None:
                            imageio.imwrite(
                                filename_template.format(ensemble=ensemble, timestep=timestep, camera=cam),
                                LoadedModel.convert_image(current_image))
                        if do_ssim:
                            ssimX[ensemble_array_idx, timestep_array_idx, cam] = ssim_loss(current_image, reference_image).item()
                        if do_lpips:
                            lpipsX[ensemble_array_idx, timestep_array_idx, cam] = lpips_loss(current_image, reference_image).item()
                        pbar.update(1)
        return {
            'min_timestep': ln.min_timestep(),
            'max_timestep': ln.max_timestep(),
            'min_ensemble': ln.min_ensemble(),
            'max_ensemble': ln.max_ensemble(),
            'num_cameras': num_cameras,
            'timings': timingsX.tolist(),
            'ssim': ssimX.tolist(),
            'lpips': lpipsX.tolist()
        }

    # load networks
    def load_and_save(cfg):
        _, filename, output_name = get_args_and_hdf5_file(cfg)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            return None, None
        try:
            ln = LoadedModel(filename, force_config_file=cfg[0][0],
                             grid_encoding=pyrenderer.SceneNetwork.LatentGrid.ByteLinear)
            if enable_preintegration:
                ln.get_image_evaluator().ray_evaluator.convert_to_texture_tf()
                ln.enable_preintegration(True)
            else:
                ln.enable_preintegration(False)
            ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
            return ln, output_name
        except Exception as e:
            print("Unable to load '%s':"%filename, e)
            return None, None

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "images_"+config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        # collect models
        lns = dict()
        base_ln = None
        configs_filtered = [cfg for cfg in configs if cfg[0]==config[1]]
        for cfg in configs_filtered:
            ln, name = load_and_save(cfg)
            lns[name] = ln
            if base_ln is None: base_ln = ln

        # render reference
        image_folder_reference = os.path.join(image_folder, "reference")
        os.makedirs(image_folder_reference, exist_ok=True)
        print("\n===================================== Render reference", cfg_index)
        cameras = base_ln.get_rotation_cameras(num_cameras)
        reference_images = render_reference_images(
            base_ln, os.path.join(image_folder_reference, IMAGE_PATTERN)
        )

        # render networks
        network_combinations = configs_filtered
        grid_only_combinations = volumetricFeaturesX + [("G%dC1"%config[2], None, (config[2], 1))] # base resolution
        for network_idx, (name, ln) in enumerate(lns.items()):
            if ln is None:
                print("\n===================================== (% 2d/% 2d) Skip network (network is None)" % (network_idx + 1, len(network_combinations)+len(grid_only_combinations)), name)
                continue
            image_folder_network = os.path.join(image_folder, "%s" % name)
            os.makedirs(image_folder_network, exist_ok=True)
            if os.path.exists(os.path.join(image_folder_network, 'stats.json')):
                print("\n===================================== (% 2d/% 2d) Skip network (already done)" % (network_idx + 1, len(network_combinations)+len(grid_only_combinations)), name)
                with open(os.path.join(image_folder_network, 'stats.json'), 'r') as f:
                    network_stats = json.load(f)
            else:
                print("\n===================================== (% 2d/% 2d) Render network" % (network_idx + 1, len(network_combinations)+len(grid_only_combinations)), name)
                network_stats = compute_stats(
                    ln, rendering_mode, reference_images, STEPSIZE,
                    os.path.join(image_folder_network, IMAGE_PATTERN),
                    True, True)
                with open(os.path.join(image_folder_network, 'stats.json'), 'w') as f:
                    json.dump(network_stats, f)
            local_stats[name] = network_stats

        # grid only for reference
        for idx, volumetricFeatures in enumerate(grid_only_combinations):
            name = "TimeVolumetricLatentSpace2-%s-l0c0-%s" % (
                config[0], volumetricFeatures[0])
            image_folder_network = os.path.join(image_folder, "%s" % name)
            os.makedirs(image_folder_network, exist_ok=True)
            if os.path.exists(os.path.join(image_folder_network, 'stats.json')):
                print("\n===================================== (% 2d/% 2d) Skip network (already done)" % (
                    len(network_combinations) + idx + 1, len(network_combinations) + len(grid_only_combinations)), name)
                with open(os.path.join(image_folder_network, 'stats.json'), 'r') as f:
                    network_stats = json.load(f)
            else:
                print("\n===================================== (% 2d/% 2d) Render network" % (
                    len(network_combinations) + idx + 1, len(network_combinations) + len(grid_only_combinations)), name)
                # load grids at keyframes
                Tkeyframes = getRange(config, '--time_keyframes')
                gridSize, gridChannels = volumetricFeatures[2]
                memory = gridChannels * (gridSize ** 3)
                target_resolution = int(np.round(np.cbrt(memory)))
                # TODO: not supported for ensembles yet
                new_volume = None
                xp = []
                fp = []
                grids = []
                def getKeyframe(t:int, return_volume):
                    print("t=", t)
                    image_evaluator, _, _ = base_ln.get_input_data().image_evaluator(
                        0, t, 0, mode='val', timestep_and_ensemble_is_actual=True)
                    print("evaluator:", image_evaluator)
                    volume = image_evaluator.volume.volume()
                    scaled_volume = volume.create_scaled(target_resolution, target_resolution, target_resolution)
                    data = scaled_volume.get_feature(0).get_level(0).to_tensor()
                    print(f"Data: shape={data.shape}, min={data[-1, ...].min().item()}, max={data[-1, ...].max().item()}")
                    return data, scaled_volume if return_volume else None
                for i,t in enumerate(Tkeyframes):
                    data,v = getKeyframe(t, i==0)
                    if i==0: new_volume = v
                    grids.append(data)
                    xp.append(t)
                    fp.append(i)
                # interpolation function
                def render_interpolated(ensemble:int, timestep:int, cam:int, timer:pyrenderer.GPUTimer) -> torch.Tensor:
                    assert ensemble==0, "ensembles not yet supported in baseline interpolation"
                    # set interpolated volume
                    t = np.interp(timestep, xp, fp)
                    ilow = np.clip(int(np.floor(t)), 0, len(grids)-1)
                    ihigh = min(ilow+1, len(grids)-1)
                    tfrac = t-ilow
                    data = (1-tfrac) * grids[ilow] + tfrac * grids[ihigh]
                    new_volume.get_feature(0).get_level(0).from_tensor(data)
                    # reset GPU, forces a memcopy from the modified CPU memory to the GPU
                    new_volume.get_feature(0).delete_all_mipmaps()
                    new_volume.get_feature(0).clear_gpu_resources()
                    # render
                    image_evaluator = base_ln.get_input_data().default_image_evaluator()
                    image_evaluator.volume.setSource(new_volume, 0)
                    image_evaluator.camera.set_parameters(cameras[cam])
                    #image_evaluator.ray_evaluator.stepsizeIsObjectSpace = False
                    image_evaluator.ray_evaluator.stepsize = STEPSIZE
                    if timer is not None:
                        timer.start()
                    img = image_evaluator.render(width, height)
                    img = image_evaluator.extract_color(img)
                    if timer is not None:
                        timer.stop()
                    return img.detach()

                network_stats = compute_stats(
                    base_ln, rendering_mode, reference_images, STEPSIZE,
                    os.path.join(image_folder_network, IMAGE_PATTERN),
                    True, True, eval_f = render_interpolated)
                with open(os.path.join(image_folder_network, 'stats.json'), 'w') as f:
                    json.dump(network_stats, f)
            local_stats[name] = network_stats

        output_stats.append(local_stats)

    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file, "w") as f:
        json.dump(output_stats, f)
    return statistics_file



def test_video():
    folder = os.path.join(BASE_PATH, 'images_ejecta-time')
    network = "TimeVolumetricLatentSpace2-ejecta-time-l32x4-fNeRF22-G32C16-none"
    with open(os.path.join(folder, network, "stats.json"), "r") as f:
        stats = json.load(f)
    min_timestep = stats['min_timestep']
    max_timestep = stats['max_timestep']
    ensemble = 0
    camera = 0

    image_size = 512
    plt_height = 200
    plt_width = 1000
    plt_dpi = 96

    #prepare LPIPS plot
    X = np.arange(min_timestep, max_timestep+1)
    Y = [stats['lpips'][ensemble][t][camera] for t in range(len(X))]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(plt_width/plt_dpi, plt_height/plt_dpi), dpi=plt_dpi)
    ax = plt.gca()
    timeline = ax.axvline(x=min_timestep, linestyle="-", color="red")
    ax.plot(X, Y, 'o-')
    Xkeyframes = [x for i,x in enumerate(X) if i%5==0]
    Ykeyframes = [y for i,y in enumerate(Y) if i%5==0]
    ax.plot(Xkeyframes, Ykeyframes, 'bo')
    ax.set_xlabel("Time")
    ax.set_ylabel("LPIPS")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    def generate_plot(t:int):
        timeline.set_xdata([t,t])
        buf = io.BytesIO()
        plt.savefig(buf, format='png', transparent=True, facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        im = Image.open(buf)
        im.load()
        buf.close()
        return im

    fnt = ImageFont.truetype("arial.ttf", 20)
    for t in range(min_timestep, max_timestep+1):
        full_image = Image.new('RGB', (2*image_size, image_size+plt_height), color="black")
        img_name = IMAGE_PATTERN.format(ensemble=ensemble, timestep=t, camera=camera)
        reference = Image.open(os.path.join(folder, "reference", img_name))
        prediction = Image.open(os.path.join(folder, network, img_name))
        full_image.paste(reference, box=(0,0))
        full_image.paste(prediction, box=(image_size,0))
        im = generate_plot(t)
        full_image.paste(im, box=((2*image_size-plt_width)//2, image_size))
        draw = ImageDraw.Draw(full_image)
        def draw_centered(text, x, y):
            w,h = fnt.getsize(text)
            draw.text((x-w//2, y), text, font=fnt)
        draw_centered("Reference", image_size//2, 5)
        draw_centered("Prediction", image_size+image_size//2, 5)
        del draw
        full_image.save(os.path.join(folder, "frame%d.png"%(t-min_timestep)))

def make_plots(statistics_file):
    print("\n===================================== Make Plots")
    with open(statistics_file, "r") as f:
        stats = json.load(f)
    output_folder = os.path.split(statistics_file)[0]
    FILETYPE = "eps"

    plt_height = 300
    plt_width = 1000
    plt_dpi = 96

    ensemble = 0

    def invLogForward(x):
        return 1-np.log(1-np.clip(x, a_min=0, a_max=0.999))
    def invLogInverse(y):
        return 1-np.exp(1-y)

    def set_scale_ssim(ax):
        ax.set_yscale('function', functions=(invLogForward, invLogInverse))
        ax.set_yticks([1.0, 0.99, 0.9, 0.8, 0.5], minor=False)
        ax.set_yticks([0.99 + 0.001*i for i in range(10)] + [0.9 + 0.01*i for i in range(10)] + [0.5 + 0.1*i for i in range(5)], minor=True)

    def set_scale_lpips(ax):
        ax.set_yscale('symlog', linthresh=2*1e-3)

    statNames = ['SSIM $\\rightarrow$', 'LPIPS $\\leftarrow$']
    statTags = ["ssim", "lpips"]
    statScales = [set_scale_ssim, set_scale_lpips]
    numStats = len(statTags)

    volumetricFeaturesX_filtered = volumetricFeaturesX[-1:] # just largest grid to avoid clutter

    for configIdx, config in enumerate(configX):
        localStats = stats[configIdx]
        networkKeys = []
        #basic
        for network, fourier, volumetricFeatures, time in itertools.product(
                networkX, fourierX, volumetricFeaturesX_filtered, timeX):
            filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
                config[0], network[0], fourier[0], volumetricFeatures[0], time[0])
            #human_name = f"Net.: {network[1]}^{network[2]}, Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, Time: {time[0]}"
            human_name = f"Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, Time: {time[0]}"
            networkKeys.append((filename, human_name))
        #steady grid
        for network, fourier, volumetricFeatures in itertools.product(
                networkX, fourierX, volumetricFeaturesX_filtered):
            filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
                config[0], network[0], fourier[0], volumetricFeatures[0], "steady-direct")
            #human_name = f"Net.: {network[1]}^{network[2]}, Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, Time: {time[0]}"
            human_name = f"Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, fixed in time"
            networkKeys.append((filename, human_name))
        # Neurcomp
        for network, fourier, volumetricFeatures in itertools.product(
                networkX, fourierX, volumetricFeaturesX_filtered):
            filename = "TimeVolumetricLatentSpace2-%s-%s-%s-%s-%s" % (
                config[0], network[0], fourier[0], volumetricFeatures[0], "LuBerger-LuBerger")
            # human_name = f"Net.: {network[1]}^{network[2]}, Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, Time: {time[0]}"
            human_name = f"neurcomp ~ ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}"
            networkKeys.append((filename, human_name))
        #only grid
        grid_only_combinations = volumetricFeaturesX_filtered + [("G%dC1"%config[2], None, (config[2], 1))] # base resolution
        for volumetricFeatures in grid_only_combinations:
            filename = "TimeVolumetricLatentSpace2-%s-l0c0-%s" % (
                config[0], volumetricFeatures[0])
            human_name = f"Grid: ${volumetricFeatures[2][0]}^3$*{volumetricFeatures[2][1]}, no network"
            networkKeys.append((filename, human_name))

        fig, axs = plt.subplots(nrows=numStats, ncols=1, sharex=True,
                                figsize=(plt_width / plt_dpi, numStats * plt_height / plt_dpi), dpi=plt_dpi)

        min_timestep = localStats[networkKeys[0][0]]['min_timestep']
        max_timestep = localStats[networkKeys[0][0]]['max_timestep']
        num_cameras = localStats[networkKeys[0][0]]['num_cameras']
        X = list(range(min_timestep, max_timestep + 1))
        Xkeyframes = list(getRange(config, '--time_keyframes'))
        Xtrain = list(getRange(config, '--time_train'))
        #XkeyframeIndices = [X.index(i) for i in Xkeyframes]
        #XtrainIndices = [X.index(i) for i in Xtrain]

        handles = OrderedDict()
        for row, (ax, statName, statTag, statScale) in enumerate(zip(axs, statNames, statTags, statScales)):
            if row==len(statNames)-1:
                ax.set_xlabel("Time")
            ax.set_ylabel(statName)
            for x in Xtrain:
                ax.axvline(x, ls='--', lw=1, color='gray')
            for x in Xkeyframes:
                ax.axvline(x, ls='-', lw=1.2, color='gray')
            for network_tag, network_name in networkKeys:
                Y = np.array([np.mean([localStats[network_tag][statTag][ensemble][t][c] for c in range(num_cameras)]) for t in range(len(X))])
                h = ax.plot(X, Y, 'o-', linewidth=1, markersize=2)
                if row == 0:
                    handles[network_name] = h
                #Ytrain = Y[XtrainIndices]
                #Ykeyframes = Y[XkeyframeIndices]
                #lx = ax.plot(Xtrain, Ytrain, 'bo')
                #ax.plot(Xkeyframes, Ykeyframes, 'bo', markersize=lx[0].get_markersize()*1.5)
            statScale(ax)

        handles = list(handles.items())
        lgd = fig.legend(
            map(lambda x: x[1][0], handles), map(lambda x: x[0], handles),
            # bbox_to_anchor=(0.75, 0.7), loc='lower center', borderaxespad=0.
            loc='center left', bbox_to_anchor=(0.9, 0.5),
            ncol=1)
        fig.subplots_adjust(hspace=+0.05)
        figure_path = os.path.join(output_folder, 'TimeVolumetricFeatures-%s.%s'%(config[0], FILETYPE))
        fig.savefig(figure_path, bbox_inches='tight', bbox_extra_artists=(lgd,))
        print("Figure saved to", figure_path)

        # copy files
        start_time = 60
        end_time = 70
        best_key = f"TimeVolumetricLatentSpace2-{config[0]}-l32x4-fNeRF14-G32C16-direct"
        #worst_key = f"TimeVolumetricLatentSpace2-{config[0]}-l32x4-fNeRF14-G32C16-both"
        neurcomp_key = f"TimeVolumetricLatentSpace2-{config[0]}-l32x4-fNeRF14-G32C16-LuBerger-LuBerger"
        network_only_key = f"TimeVolumetricLatentSpace2-{config[0]}-l0c0-G{config[2]}C1"
        images_out_folder = os.path.join(output_folder, 'images-out')
        os.makedirs(images_out_folder, exist_ok=True)

        def calcCropAndSave(files_in, files_out):
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

        def useCropAndSave(files_in, files_out, crop):
            minX, minY, maxX, maxY = crop
            for fi, fo in zip(files_in, files_out):
                img = imageio.imread(fi)
                img = img[minX:maxX, minY:maxY, :]
                imageio.imwrite(fo, img)

        files_in = []
        files_out = []
        for t in range(start_time, end_time+1):
            input_filename = IMAGE_PATTERN.format(ensemble=ensemble, timestep=t, camera=0)
            files_in.append(os.path.join(output_folder, "images_%s/reference/%s" % (config[0], input_filename)))
            files_out.append(os.path.join(images_out_folder, "%s_%03d_reference.png" % (config[0], t)))
        crop = calcCropAndSave(files_in, files_out)

        files_in = []
        files_out = []
        for t in range(start_time, end_time + 1):
            input_filename = IMAGE_PATTERN.format(ensemble=ensemble, timestep=t, camera=0)
            files_in.append(os.path.join(output_folder, "images_%s/%s/%s" % (config[0], best_key, input_filename)))
            files_out.append(os.path.join(images_out_folder, "%s_%03d_best.png" % (config[0], t)))
            files_in.append(os.path.join(output_folder, "images_%s/%s/%s" % (config[0], neurcomp_key, input_filename)))
            files_out.append(os.path.join(images_out_folder, "%s_%03d_neurcomp.png" % (config[0], t)))
            files_in.append(os.path.join(output_folder, "images_%s/%s/%s" % (config[0], network_only_key, input_filename)))
            files_out.append(os.path.join(images_out_folder, "%s_%03d_no_network.png" % (config[0], t)))
        useCropAndSave(files_in, files_out, crop)

        # create latex table
        IMAGE_PREFIX = ""
        with open(os.path.join(images_out_folder, "%s-table.tex"%config[0]), "w") as f:
            f.write("""
\\documentclass[10pt,a4paper]{standalone}
\\usepackage{graphicx}
\\begin{document}

\\newcommand{\\timesize}{0.2}%
\\setlength{\\tabcolsep}{0pt}%
""")
            f.write("\\begin{tabular}{%s}%%\n"%("c"*(end_time-start_time+2)))
            for rowName, rowTag in zip(["a)", "b)", "c)", "d)"], ["reference", "best", "neurcomp", "no_network"]):
                f.write("\t%s"%rowName)
                for t in range(start_time, end_time + 1):
                    img_name = IMAGE_PREFIX + "%s_%03d_%s.png" % (config[0], t, rowTag)
                    f.write(" &%%\n\t\\raisebox{-.5\\height}{\\includegraphics[width=\\timesize\\textwidth]{%s}}"%img_name)
                f.write("\\\\%\n")
            f.write("\t")
            for t in range(start_time, end_time + 1):
                f.write(" & %d"%t)
            f.write("\\\\%\n")
            f.write("\\end{tabular}%\n")
            f.write("\\end{document}")

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()