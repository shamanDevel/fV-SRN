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
import matplotlib.pyplot as plt
import matplotlib.ticker
from collections import defaultdict

BEST_ACTIVATION = "SnakeAlt:1"
BEST_NETWORK = (32,4)
GRID_RESOLUTION = 32
GRID_CHANNELS = 16

BASE_PATH = 'volnet/results/eval_ImportanceGrid'
FILENAME_PATTERN = "importance-world-%s-%s-%s"

configX = [
    ("plume100", "config-files/plume100-v2-dvr.json"),
    ("ejecta70", "config-files/ejecta70-v6-dvr.json"),
    ("RM20", "config-files/RichtmyerMeshkov-t20-v1-dvr.json"),
    ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json"),
]
importanceX = [
        ("i001", "0.01", ["--train:sampler_importance", "0.01"]),
        ("i002", "0.02", ["--train:sampler_importance", "0.02"]),
        ("i005", "0.05", ["--train:sampler_importance", "0.05"]),
        ("i01", "0.1", ["--train:sampler_importance", "0.1"]),
        ("i02", "0.2", ["--train:sampler_importance", "0.2"]),
        ("i05", "0.5", ["--train:sampler_importance", "0.5"]),
        ("i1", "off", []),
        ("iRebuildDensity", "density", ["--train:sampler_importance", "0.01", '--rebuild_dataset', '51']),
        ("iRebuildColor", "color", ["--train:sampler_importance", "0.01", '--rebuild_dataset', '51', '--rebuild_force_color']),
    ]
numExtraImportance = 2 # rebuild-samplers
fourierX = [("fNeRF", -1)]

def main():
    configs = collect_configurations()
    train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    for config, fourier, importance in itertools.product(configX, fourierX, importanceX):
        filename = FILENAME_PATTERN % (
            config[0], fourier[0], importance[0])
        cfgs.append((config[1], fourier[1], importance[2], filename))
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
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "--activation", BEST_ACTIVATION,
        "-l1", "1",
        "--lr_step", "50",
        "-i", "200",
        "--activation", BEST_ACTIVATION,
        "--layers", ':'.join([str(BEST_NETWORK[0])] * (BEST_NETWORK[1] - 1)),
        "--volumetric_features_resolution", str(GRID_RESOLUTION),
        "--volumetric_features_channels", str(GRID_CHANNELS),
        "--logdir", BASE_PATH+'/log',
        "--modeldir", BASE_PATH+'/model',
        "--hdf5dir", BASE_PATH+'/hdf5',
    ]

    def getFourierParameters(fourier):
        std = fourier
        return ['--fouriercount', str((BEST_NETWORK[0] - 4) // 2), '--fourierstd', str(std)]

    config, fourier, importance, filename = cfg

    launcher = [sys.executable, "volnet/train_volnet.py"]
    args = launcher + [config] + \
           common_parameters + \
           getFourierParameters(fourier) + \
           importance + ['--name', filename]

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

    num_cameras = 64
    width = 512
    height = 512
    STEPSIZE = 1/512
    timer = pyrenderer.GPUTimer()
    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    #rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = rendering_mode==LoadedModel.EvaluationMode.TENSORCORES_MIXED

    output_stats = []
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln, mode, reference_images, stepsize, filename_template=None, do_ssim=False, do_lpips=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            current_image = ln.render_network(
                cameras[i], width, height, mode,
                stepsize, timer=timer)
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
    def load_and_save(cfg):
        _, filename, output_name = get_args_and_hdf5_file(cfg)
        ln = LoadedModel(filename)
        if enable_preintegration:
            ln.enable_preintegration(True)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        return ln, output_name

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "images_"+config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        reference_images = None
        # collect models
        lns = dict()
        base_ln = None
        for fourier, importance in itertools.product(fourierX, importanceX):
            filename = FILENAME_PATTERN % (
                config[0], fourier[0], importance[0])
            ln, name = load_and_save((config[1], fourier[1], importance[2], filename))
            lns[(fourier[0], importance[0])] = (ln, name)
            if base_ln is None: base_ln = ln

        # render reference
        if reference_images is None:
            image_folder_reference = os.path.join(image_folder, "reference")
            os.makedirs(image_folder_reference, exist_ok=True)
            print("\n===================================== Render reference", cfg_index)
            cameras = base_ln.get_rotation_cameras(num_cameras)
            reference_images = [None] * num_cameras
            for i in range(num_cameras):
                reference_images[i] = base_ln.render_reference(cameras[i], width, height)
                imageio.imwrite(
                    os.path.join(image_folder_reference, 'reference%03d.png' % i),
                    LoadedModel.convert_image(reference_images[i]))

        # render networks
        for fourier, importance in itertools.product(fourierX, importanceX):
            ln, name = lns[(fourier[0], importance[0])]
            image_folder_screen = os.path.join(image_folder, "%s" % name)
            os.makedirs(image_folder_screen, exist_ok=True)
            time, ssim, lpips = compute_stats(
                ln, rendering_mode, reference_images, STEPSIZE,
                os.path.join(image_folder_screen, 'img%03d.png'),
                True, True)
            local_stats[name] = {
                'time': time,
                'ssim': ssim,
                'lpips': lpips,
            }

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

    numRows = len(configX)
    statNames = ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']
    statTags = ["ssim", "lpips"]
    numCols = len(statTags)

    fourier = fourierX[0][0]

    fig, axs = plt.subplots(1, numCols, squeeze=False, sharex=True, figsize=(7, 3))
    legend_handles = []
    legend_names = []
    for row, offX in zip(range(numRows), np.linspace(-0.02, +0.02, numRows, endpoint=True)):
        local_stat = stats[row]
        #axs[row, 0].set_ylabel(configX[row][0])
        for col, (name, tag) in enumerate(zip(statNames, statTags)):
            ax = axs[0,col]
            if row==0:
                ax.set_title(name)

            X = []
            Xlabel = []
            Y = []
            err = []
            for i,(impN, impH, impV) in enumerate(importanceX[:-numExtraImportance]):
                filename = FILENAME_PATTERN % (
                    configX[row][0], fourier, impN)
                y,e = local_stat[filename][tag]
                X.append(i+offX)
                Xlabel.append(impH)
                Y.append(y)
                err.append(e)
            h = ax.errorbar(X, Y, yerr=err, fmt='.-')
            # extra
            for i,(impN, impH, impV) in enumerate(importanceX[-numExtraImportance:]):
                filename = FILENAME_PATTERN % (
                    configX[row][0], fourier, impN)
                y,e = local_stat[filename][tag]
                X.append(X[-1]+1)
                Xlabel.append(impH)
                Y.append(y)
                err.append(e)
                ax.errorbar(X[-1:], Y[-1:], yerr=err[-1:], color=h[0].get_color(), fmt='.')

            # legend
            if col==0:
                legend_handles.append(h)
                legend_names.append(configX[row][0])
            ax.set_xticks(X)
            ticks = ax.set_xticklabels(Xlabel)
            for tick in ticks: #[-numExtraImportance:]:
                tick.set_rotation(45)
            ax.set_xlabel("Importance p")

        # determine and copy best and worst images
        tag = "lpips"
        worst_lpips = 0
        worst_filename = None
        best_lpips = 1
        best_filename = None
        for i, (impN, impH, impV) in enumerate(importanceX[:-numExtraImportance]):
            filename = FILENAME_PATTERN % (
                configX[row][0], fourier, impN)
            y, _ = local_stat[filename][tag]
            if y < best_lpips:
                best_lpips = y
                best_filename = filename
            if y > worst_lpips:
                worst_lpips = y
                worst_filename = filename
        density_filenames = [FILENAME_PATTERN % (configX[row][0], fourier, importanceX[-2][0])]
        density_filename = min(density_filenames, key=lambda filename: local_stat[filename][tag][0])
        color_filenames = [FILENAME_PATTERN % (configX[row][0], fourier, importanceX[-1][0])]
        color_filename = min(color_filenames, key=lambda filename: local_stat[filename][tag][0])

        shutil.copyfile(
            os.path.join(output_folder, "images_%s/reference/reference000.png" % (configX[row][0])),
            os.path.join(output_folder, "%s_reference.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], best_filename)),
            os.path.join(output_folder, "%s_best.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], worst_filename)),
            os.path.join(output_folder, "%s_worst.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], worst_filename)),
            os.path.join(output_folder, "%s_worst.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], density_filename)),
            os.path.join(output_folder, "%s_density.png" % configX[row][0]))
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], color_filename)),
            os.path.join(output_folder, "%s_color.png" % configX[row][0]))

    lgd = fig.legend(
        legend_handles, legend_names,
        # bbox_to_anchor=(0.75, 0.7), loc='lower center', borderaxespad=0.
        loc='center left', bbox_to_anchor=(0.9, 0.5),
        ncol=1)
    fig.savefig(os.path.join(output_folder, 'ImportanceGrid-SSIM.%s'%FILETYPE),
                bbox_inches='tight', bbox_extra_artists=(lgd,))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()