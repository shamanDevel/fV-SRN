"""
Script to evaluate the network quality if it is trained for densities or for colors.
They are tested with world-space training and the best configuration from eval_network_configs
"""

import numpy as np
import sys
import os
import subprocess
import itertools
import imageio
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker

BEST_ACTIVATION = "SnakeAlt:1"
BEST_NETWORK = (32,4)
GRID_RESOLUTION = 32
GRID_CHANNELS = 16

CONFIG_FILES = [
"config-files/plumeEnsemble-v0-dvr.json",
"config-files/plumeEnsemble-v1-dvr.json",
"config-files/plumeEnsemble-v2-dvr.json",
]
VOLUME_FILES = "volumes/ScalarFlow/sim_{ensemble:06d}/volume_{time:06d}.cvol"
VOLUME_TIMESTEP = 100

def main():
    configs = collect_configurations()
    train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    return [
        ("density", 0),
        ("rgbo", 0),
        ("rgbo", 1),
        ("rgbo", 2),
    ]

def get_args_and_hdf5_file(output_mode, config_file_index:int):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :param activation: the activation function name
    :param network: the network combination (channels, layers)
    :return: args, filename
    """
    output_name = "run_%s_%d"%(output_mode, config_file_index)
    parameters = [
        sys.executable, "volnet/train_volnet.py",
        CONFIG_FILES[config_file_index],
        "--volume_filenames",
        VOLUME_FILES,
        "--time_keyframes", f"{VOLUME_TIMESTEP}:{VOLUME_TIMESTEP+1}:1",
        "--time_train", f"{VOLUME_TIMESTEP}:{VOLUME_TIMESTEP+1}:1",
        "--time_val", f"{VOLUME_TIMESTEP}:{VOLUME_TIMESTEP+1}:1",
        "--train:mode", "world",
        "--train:samples", "256**3",
        "--train:batchsize", "64*64*128",
        "--train:sampler_importance", "0.01",
        '--rebuild_dataset', '51',
        "--val:copy_and_split",
        "--outputmode", "%s:direct"%output_mode,
        "--lossmode", "%s"%output_mode,
        "-l1", "1",
        "--lr_step", "100",
        "-i", "200",
        '--fouriercount', str((BEST_NETWORK[0]-4)//2), '--fourierstd', '1.0',
        "--activation", BEST_ACTIVATION,
        "--layers", ':'.join([str(BEST_NETWORK[0])]*(BEST_NETWORK[1]-1)),
        "--volumetric_features_resolution", str(GRID_RESOLUTION),
        "--volumetric_features_channels", str(GRID_CHANNELS),
        "--logdir", 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/log',
        "--modeldir", 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/model',
        "--hdf5dir", 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/hdf5',
        '--name', output_name,
        '--save_frequency', '50'
    ]
    hdf5_file = 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/hdf5/' + output_name + ".hdf5"
    return parameters, hdf5_file

def train(configs):
    print("Configurations:", len(configs))
    for output_mode, config_file_index in configs:
        args, filename = get_args_and_hdf5_file(output_mode, config_file_index)
        if os.path.exists(filename):
            print("Skipping test", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval(configs):
    print("Evaluate")
    statistics_file = 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/stats.json'
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    num_cameras = 64
    width = 512
    height = 512
    stepsize = 1 / 512
    timer = pyrenderer.GPUTimer()

    output_stats = []
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln, mode, reference_images, filename_template=None, do_ssim=False, do_lpips=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            current_image = ln.render_network(
                cameras[i], width, height, mode,
                stepsize, timer=timer, timestep=VOLUME_TIMESTEP)
            if i>0:
                timingsX.append(timer.elapsed_milliseconds())
            if filename_template is not None:
                imageio.imwrite(
                    filename_template%i,
                    LoadedModel.convert_image(current_image))
            if do_ssim:
                ssimX.append(ssim_loss(current_image, reference_images[i]).item())
            if do_lpips:
                lpipsX.append(lpips_loss(current_image, reference_images[i]).item())
        return \
            (np.mean(timingsX), np.std(timingsX)), \
            (np.mean(ssimX), np.std(ssimX)) if do_ssim else (np.NaN, np.NaN), \
            (np.mean(lpipsX), np.std(lpipsX)) if do_lpips else (np.NaN, np.NaN)

    # load networks
    def load_and_save(output_mode, index):
        _, filename = get_args_and_hdf5_file(output_mode, index)
        ln = LoadedModel(filename)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        return ln
    ln_density = load_and_save('density', 0)
    ln_colors = []
    for i in range(len(CONFIG_FILES)):
        ln_colors.append(load_and_save('rgbo', i))

    for cfg_idx, cfg in enumerate(CONFIG_FILES):
        image_folder = 'volnet/results/eval_world_DensityVsColorGrid_WithResampling/images_%d/'%(cfg_idx)
        os.makedirs(image_folder, exist_ok=True)
        local_stats = {
            'cfg_index': cfg_idx,
            'cfg': cfg}
        # render reference
        print(cfg)
        print("\n===================================== Render reference")
        ln = ln_colors[cfg_idx]
        cameras = ln.get_rotation_cameras(num_cameras)
        reference_images = [None] * num_cameras
        for i in range(num_cameras):
            reference_images[i] = ln.render_reference(cameras[i], width, height,
                                                      timestep=VOLUME_TIMESTEP)
            imageio.imwrite(
                os.path.join(image_folder, 'reference%03d.png' % i),
                LoadedModel.convert_image(reference_images[i]))

        # render color network
        time, ssim, lpips = compute_stats(
            ln, LoadedModel.EvaluationMode.TENSORCORES_MIXED, reference_images,
            os.path.join(image_folder, 'color%03d.png'),
            True, True)
        local_stats['color-time'] = time
        local_stats['color-ssim'] = ssim
        local_stats['color-lpips'] = lpips

        # render density network
        ln.set_network_pytorch(*ln_density.get_network_pytorch())
        ln.set_network_tensorcores(ln_density.get_network_tensorcores())
        time, ssim, lpips = compute_stats(
            ln, LoadedModel.EvaluationMode.TENSORCORES_MIXED, reference_images,
            os.path.join(image_folder, 'density%03d.png'),
            True, True)
        local_stats['density-time'] = time
        local_stats['density-ssim'] = ssim
        local_stats['density-lpips'] = lpips

        output_stats.append(local_stats)

    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file, "w") as f:
        json.dump(output_stats, f)
    return statistics_file

def make_plots(statistics_file):
    print("\n===================================== Make Plots")
    with open(statistics_file, "r") as f:
        stats = json.load(f)
    output_folder = os.path.split(statistics_file)[0]
    FILETYPE = "eps"

    numCols = len(stats)
    numClasses = 2
    fig, axs = plt.subplots(1, 2, sharex=True, figsize=(6.4, 3.0))
    x_offset = np.linspace(-0.2, +0.2, numClasses, True)
    width = x_offset[1] - x_offset[0]
    handles = []
    for ax, stat, label in zip(axs, ["ssim", "lpips"], ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']):
        for i, cls in enumerate(["density", "color"]):
            X = []
            Y = []
            err = []
            for j in range(numCols):
                X.append(j+x_offset[i])
                y,e = stats[j]["%s-%s"%(cls, stat)]
                Y.append(y)
                err.append(e)
            h = ax.bar(X, Y, width=width, yerr=err)
            ax.bar_label(h, label_type='center', fmt="%.3f", fontsize=7)
            if stat=='ssim':
                handles.append(h)
        ax.set_title(label)
        ax.set_xticks(np.arange(numCols))
        ax.set_xticklabels(["TF %d"%(i+1) for i in range(numCols)])

    lgd = fig.legend(
        handles, ["Density", "Color"],
        bbox_to_anchor=(0.65, 0.7), loc='lower center', borderaxespad=0.)
    fig.savefig(os.path.join(output_folder, 'DensityVsColorGrid-WithResampling-SSIM.%s'%FILETYPE),
                bbox_inches='tight', bbox_extra_artists=(lgd,))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()