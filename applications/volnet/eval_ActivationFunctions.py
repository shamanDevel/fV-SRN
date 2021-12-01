"""
Script to evaluate the activation functions for the selected network + grid.
"""


import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import sys
import os
import subprocess
import itertools
import imageio
import json
import torch
import io
import shutil
import matplotlib.pyplot as plt
import matplotlib.ticker
from collections import defaultdict
from typing import Tuple

# From eval_VolumetricFeatures.py / Section 5.2
BEST_NETWORK = (32,4)
GRID_RESOLUTION = 32
GRID_CHANNELS = 16

activationX = ["ReLU", "Sine", "Snake:2", "SnakeAlt:1"]

BASE_PATH = 'volnet/results/eval_ActivationFunctions'

configX = [
        ("plume100", "config-files/plume100-v2-dvr.json", "Plume"),
        ("ejecta70", "config-files/ejecta70-v6-dvr.json", "Ejecta"),
        ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json", "RM"),
        #("Skull5", "neuraltextures/config-files/skull-v5-dvr.json"),
    ]


def main():
    train()
    statistics_file = eval()
    make_plots(statistics_file)

def get_args_and_hdf5_file(activation, config: Tuple[str, str, str]):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :param activation: the activation function name
    :param network: the network combination (channels, layers)
    :return: args, filename
    """
    config_name, config_settings, human_name = config

    output_name = "run_%s_%s"%(config_name, activation.replace(':','-'))
    parameters = [
        sys.executable, "volnet/train_volnet.py",
        config_settings,
        "--train:mode", "world",
        "--train:samples", "256**3",
        "--train:batchsize", "64*64*128",
        "--train:sampler_importance", "0.01",
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "-l1", "1",
        "--lr_step", "50",
        "-i", "200",
        '--fouriercount', str((BEST_NETWORK[0]-4)//2), '--fourierstd', '1.0',
        "--activation", activation,
        "--layers", ':'.join([str(BEST_NETWORK[0])]*(BEST_NETWORK[1]-1)),
        "--volumetric_features_resolution", str(GRID_RESOLUTION),
        "--volumetric_features_channels", str(GRID_CHANNELS),
        "--logdir", BASE_PATH+'/log',
        "--modeldir", BASE_PATH+'/model',
        "--hdf5dir", BASE_PATH+'/hdf5',
        '--name', output_name,
        '--save_frequency', '50'
    ]
    hdf5_file = BASE_PATH+'/hdf5/' + output_name + ".hdf5"
    return parameters, hdf5_file, output_name

def train():
    print("Configurations:", len(activationX) * len(configX))
    for config in configX:
        for activation in activationX:
            args, filename, _ = get_args_and_hdf5_file(activation, config)
            if os.path.exists(filename):
                print("Skipping test", filename)
            else:
                print("\n=====================================\nRun", filename)
                subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval():
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
    STEPSIZE = 1 / 512
    timer = pyrenderer.GPUTimer()

    if os.name != 'nt':
        rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    else:
        rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    enable_preintegration = True

    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln: LoadedModel, mode, reference_images, stepsize, filename_template=None,
                      do_ssim=False, do_lpips=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            if enable_preintegration:
                ln.enable_preintegration(True, convert_to_texture=True)
            else:
                ln.enable_preintegration(False)
            current_image = ln.render_network(
                cameras[i], width, height, mode,
                stepsize, timer=timer)
            if i > 0:
                timingsX.append(timer.elapsed_milliseconds())
            if filename_template is not None:
                imageio.imwrite(
                    filename_template % i,
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
    def load_and_save(activation, config):
        _, filename, output_name = get_args_and_hdf5_file(activation, config)
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            return None, None
        try:
            ln = LoadedModel(filename)
            # if enable_preintegration:
            #    ln.enable_preintegration(True)
            ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
            return ln, output_name
        except Exception as e:
            print("Unable to load '%s':" % filename, e)
            return None, None

    output_stats = {}

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "images_" + config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        reference_images = None
        # collect models
        lns = dict()
        base_ln = None
        for activation in activationX:
            ln, name = load_and_save(activation, config)
            lns[activation] = (ln, name)
            if base_ln is None: base_ln = ln

        # render reference
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
        for activation in activationX:
            ln, name = lns[activation]
            if ln is None:
                print("Skip", name, ", network is None")
                continue
            print("Render", name)
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

        output_stats[config[0]] = local_stats

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

    statNames = ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']
    statTags = ["ssim", "lpips"]
    statAggregation = [max, min]

    latex = io.StringIO()
    # latex header
    latex.write("\\begin{tabular}{r%s}\n" % ("cc" * (len(configX))))
    latex.write("\\toprule\n")
    latex.write("\\multirow{2}{*}{Activation}")
    for config in configX:
        latex.write(" & \\multicolumn{2}{c}{%s}"%config[2])
    latex.write("\\\\\n")
    for config in configX:
        latex.write(" & %s & %s" % tuple(statNames))
    latex.write("\\\\\n")
    latex.write("\n\\midrule\n")

    best_per_dataset = dict()
    for config in configX:
        cfg_index = stats[config[0]]['cfg_index']
        for tag, aggr in zip(statTags, statAggregation):
            values = []
            for activation in activationX:
                _, _, n = get_args_and_hdf5_file(activation, configX[cfg_index])
                v = "%.4f" % stats[config[0]][n][tag][0]
                values.append(v)
            best_per_dataset[(cfg_index, tag)] = aggr(values)


    # main content
    for activation in activationX:
        latex.write(activation.split(':')[0])
        for config in configX:
            cfg_index = stats[config[0]]['cfg_index']
            _, _, n = get_args_and_hdf5_file(activation, configX[cfg_index])
            for tag in statTags:
                v = "%.4f"%stats[config[0]][n][tag][0]
                if v == best_per_dataset[(cfg_index, tag)]:
                    latex.write(" & $\\bm{%s}$"%v)
                else:
                    latex.write(" & $%s$" % v)
        latex.write("\\\\\n")

    #footer
    latex.write("\n\\bottomrule\n")
    latex.write("\\end{tabular}\n")

    latex = latex.getvalue()
    with open(os.path.join(output_folder, "ActivationFunctions.tex"), 'w') as f:
        f.write(latex)
    print(latex)

    print("Done")

if __name__ == '__main__':
    main()
