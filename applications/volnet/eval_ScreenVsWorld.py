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
BEST_NETWORK = (48,10)

CONFIG_FILES = [
    "config-files/plumeEnsemble-v0-dvr.json",
    "config-files/plumeEnsemble-v1-dvr.json",
    "config-files/plumeEnsemble-v2-dvr.json",
]
VOLUME_FILES = "volumes/ScalarFlow/sim_{ensemble:06d}/volume_{time:06d}.cvol"
STEPSIZES = [
    0.005,
    0.02,
    0.1
]


def main():
    configs = collect_configurations()
    #train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    cfgs.append(("world", 0, 1, 'plain'))
    for i in range(len(CONFIG_FILES)):
        for s in STEPSIZES:
            cfgs.append(("screen", i, s, 'dirD'))
            cfgs.append(("screen", i, s, 'dirF'))
            cfgs.append(("screen", i, s, 'dirS'))
            cfgs.append(("screen", i, s, 'plain'))
    return cfgs

def get_args_and_hdf5_file(training_mode, config_file_index:int, stepsize:float,
                           direction_mode: str):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :param activation: the activation function name
    :param network: the network combination (channels, layers)
    :return: args, filename
    """
    assert direction_mode in ['plain', 'dirD', 'dirF', 'dirS']
    output_name = "run_%s_%d_%d_%s"%(
        training_mode, config_file_index, stepsize*1000, direction_mode)

    if direction_mode=='dirD':
        use_direction = True
        disable_direction_in_fourier_features = True
        fourier_position_direction_split = False
    elif direction_mode=='dirF':
        use_direction = True
        disable_direction_in_fourier_features = False
        fourier_position_direction_split = False
    elif direction_mode=='dirS':
        use_direction = True
        disable_direction_in_fourier_features = False
        fourier_position_direction_split = True
    else: #if direction_mode == 'plain':
        use_direction = False
        disable_direction_in_fourier_features = False
        fourier_position_direction_split = False
    num_fourier_screen = (BEST_NETWORK[0]-(8 if use_direction else 4))//2

    parametersWorld = [
        "python", "volnet/train_volnet.py",
        CONFIG_FILES[config_file_index],
        "--volume_filenames",
        VOLUME_FILES,
        "--time_keyframes", "100:101:1",
        "--time_train", "100:101:1",
        "--time_val", "100:101:1",
        "--train:mode", "world",
        "--train:samples", "256**3",
        "--train:batchsize", "64*64*128",
        "--train:sampler_importance", "0.01",
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "-l1", "1",
        "--lr_step", "100",
        "-i", "500",
        '--fouriercount', str((BEST_NETWORK[0]-4)//2), '--fourierstd', '1.0',
        "--activation", BEST_ACTIVATION,
        "--layers", ':'.join([str(BEST_NETWORK[0])]*(BEST_NETWORK[1]-1)),
        "--logdir", 'volnet/results/eval_ScreenVsWorld/log',
        "--modeldir", 'volnet/results/eval_ScreenVsWorld/model',
        "--hdf5dir", 'volnet/results/eval_ScreenVsWorld/hdf5',
        '--name', output_name,
        '--save_frequency', '50'
    ]
    parametersScreen = [
        "python", "volnet/train_volnet.py",
        CONFIG_FILES[config_file_index],
        "--volume_filenames",
        VOLUME_FILES,
        "--time_keyframes", "100:101:1",
        "--time_train", "100:101:1",
        "--time_val", "100:101:1",
        "--train:mode", "screen",
        "--train:views", "96",
        "--train:resolution", "256",
        "--train:stepsize", str(stepsize),
        "--train:batchsize", "8",
        "--train:sampler_importance", "0.01",
        "--val:mode", "screen",
        "--val:views", "2",
        "--val:resolution", "256",
        "--val:stepsize", str(stepsize),
        "--val:batchsize", "2",
        "--vis:stepsize", str(stepsize),
        "--outputmode", "rgbo",
        "--lossmode", "rgbo",
        "-l1", "1",
        "--lr_step", "100",
        "-lr", "0.01",
        "-i", "500",
        '--fouriercount', str(num_fourier_screen), '--fourierstd', '1.0',
        "--activation", BEST_ACTIVATION,
        "--layers", ':'.join([str(BEST_NETWORK[0])] * (BEST_NETWORK[1] - 1)),
        "--logdir", 'volnet/results/eval_ScreenVsWorld/log',
        "--modeldir", 'volnet/results/eval_ScreenVsWorld/model',
        "--hdf5dir", 'volnet/results/eval_ScreenVsWorld/hdf5',
        '--name', output_name,
        '--save_frequency', '50'
    ]
    if use_direction:
        parametersScreen.append('--use_direction')
    if disable_direction_in_fourier_features:
        parametersScreen.append('--disable_direction_in_fourier_features')
    if fourier_position_direction_split:
        parametersScreen.append('--fourier_position_direction_split')
        parametersScreen.append(str(num_fourier_screen//2))
    hdf5_file = 'volnet/results/eval_ScreenVsWorld/hdf5/' + output_name + ".hdf5"
    return parametersScreen if training_mode=='screen' else parametersWorld, hdf5_file, output_name

def train(configs):
    print("Configurations:", len(configs))
    for cfg in configs:
        args, filename, outputname = get_args_and_hdf5_file(*cfg)
        if os.path.exists(filename):
            print("Skipping test", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval(configs):
    print("Evaluate")
    statistics_file = 'volnet/results/eval_ScreenVsWorld/stats.json'
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
    timer = pyrenderer.GPUTimer()
    #rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    rendering_mode = LoadedModel.EvaluationMode.PYTORCH16

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
    def load_and_save(training_mode, cfg_index, stepsize, direction_mode):
        _, filename, output_name = get_args_and_hdf5_file(training_mode, cfg_index, stepsize, direction_mode)
        ln = LoadedModel(filename)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        return ln, output_name

    ln_density, _ = load_and_save("world", 0, 1, 'plain')

    """
    cfgs.append(("world", 0, 1, False, False))
    for i in range(len(CONFIG_FILES)):
        for s in STEPSIZES:
            cfgs.append(("screen", i, s, True, False))
            cfgs.append(("screen", i, s, True, True))
            cfgs.append(("screen", i, s, False, False))
    """

    for cfg_index in range(len(CONFIG_FILES)):
        image_folder = 'volnet/results/eval_ScreenVsWorld/images_%d/'%cfg_index
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': CONFIG_FILES[cfg_index]}

        reference_images = None
        for s in STEPSIZES:
            ln_screen = []
            # collect models
            ln_screen.append(load_and_save("screen", cfg_index, s, 'plain'))
            ln_screen.append(load_and_save("screen", cfg_index, s, 'dirD'))
            ln_screen.append(load_and_save("screen", cfg_index, s, 'dirS'))
            ln_screen.append(load_and_save("screen", cfg_index, s, 'dirF'))
            # render reference
            if reference_images is None:
                image_folder_reference = os.path.join(image_folder, "reference")
                os.makedirs(image_folder_reference, exist_ok=True)
                print("\n===================================== Render reference", cfg_index)
                ln = ln_screen[0][0]
                cameras = ln.get_rotation_cameras(num_cameras)
                reference_images = [None] * num_cameras
                for i in range(num_cameras):
                    reference_images[i] = ln.render_reference(cameras[i], width, height)
                    imageio.imwrite(
                        os.path.join(image_folder_reference, 'reference%03d.png' % i),
                        LoadedModel.convert_image(reference_images[i]))

            # render screen networks
            for ln, name in ln_screen:
                image_folder_screen = os.path.join(image_folder, "%s" % name)
                os.makedirs(image_folder_screen, exist_ok=True)
                time, ssim, lpips = compute_stats(
                    ln, rendering_mode, reference_images, s,
                    os.path.join(image_folder_screen, 'screen%03d.png'),
                    True, True)
                local_stats[name] = {
                    'time': time,
                    'ssim': ssim,
                    'lpips': lpips,
                }

            # render density (world) network
            ln = ln_screen[0][0]
            image_folder_density = os.path.join(image_folder, "world_%d"%(1000*s))
            os.makedirs(image_folder_density, exist_ok=True)
            ln.set_network_pytorch(*ln_density.get_network_pytorch())
            ln.set_network_tensorcores(ln_density.get_network_tensorcores())
            time, ssim, lpips = compute_stats(
                ln, rendering_mode, reference_images, s,
                os.path.join(image_folder_density, 'world%03d.png'),
                True, True)
            local_stats['world_%d'%(1000*s)] = {
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

    latex = io.StringIO()
    LATEX_SHOW_STD = False

    numStepsizes = len(STEPSIZES)
    numTFs = len(CONFIG_FILES)
    numClasses = 5
    classNames = ["world", "pos", "dirP", "dirS", "dirF"]
    def classTags(tf_idx):
        return ["world_%d", f"run_screen_{tf_idx}_%d_plain", f"run_screen_{tf_idx}_%d_dirD", f"run_screen_{tf_idx}_%d_dirS",
            f"run_screen_{tf_idx}_%d_dirF"]
    statNames = ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']
    statTags = ["ssim", "lpips"]

    # latex header
    latex.write("\\begin{tabular}{@{}rr%s@{}}\n"%("c"*(len(statNames)*numStepsizes)))
    latex.write("\\toprule\n")
    latex.write("&")
    for j,s in enumerate(STEPSIZES):
        latex.write(" & \\multicolumn{%d}{c}{Stepsize %s}"%(len(statNames), "%.0e"%s))
    latex.write("\\\\\n")
    for j in range(len(STEPSIZES)):
        latex.write("\\cmidrule(r){%d-%d}"%(3+len(statNames)*j, 2+len(statNames)*(j+1)))
    latex.write("\n TF & Input ")
    for j in range(len(STEPSIZES)):
        for s in statNames:
            latex.write(" & %s"%s)
    latex.write("\\\\\n")

    fig, axs = plt.subplots(numTFs, 2, squeeze=False, sharex=True, figsize=(6.4, 1+2*numTFs))
    x_offset = np.linspace(-0.3, +0.3, numClasses, True)
    width = x_offset[1] - x_offset[0]
    handles = []
    handle_names = []
    for row in range(numTFs):

        # plot
        local_stat = stats[row]
        axs[row,0].set_ylabel('TF %d'%(row+1))
        for k, (ax, stat, label) in enumerate(zip(axs[row], statTags, statNames)):
            for i, (cls, tag) in enumerate(zip(classNames, classTags(row))):
                X = []
                Y = []
                err = []
                for j,s in enumerate(STEPSIZES):
                    X.append(j+x_offset[i])
                    y,e = local_stat[tag%(1000*s)][stat]
                    Y.append(y)
                    err.append(e)
                h = ax.bar(X, Y, width=width, yerr=err)
                if stat=='ssim':
                    handles.append(h)
                    handle_names.append(cls)
            Xlabel = ["%.0e"%s for s in STEPSIZES]
            ax.set_title(label)
            ax.set_xticks(np.arange(numStepsizes))
            ax.set_xticklabels(Xlabel)
            ax.set_xlabel("Stepsize")

        # latex
        latex.write("\\cmidrule(r){1-2}")
        for j in range(len(STEPSIZES)):
            latex.write("\\cmidrule(r){%d-%d}"%(3+len(statNames)*j, 2+len(statNames)*(j+1)))
        latex.write("\\multirow{%d}{*}{TF %d}" % (numClasses, row + 1))
        tags = classTags(row)
        best_ssim = dict()
        best_lpips = dict() #defaultdict(lambda: 1.0)
        for j, s in enumerate(STEPSIZES):
            best_ssim[j] = max([local_stat[tags[i] % (1000 * s)]['ssim'][0] for i in range(len(classNames))])
            best_lpips[j] = min([local_stat[tags[i] % (1000 * s)]['lpips'][0] for i in range(len(classNames))])
        best_stats = {'ssim': best_ssim, 'lpips': best_lpips}

        for i in range(len(classNames)):
            tag = tags[i]
            latex.write(" & %s"%classNames[i])
            for j,s in enumerate(STEPSIZES):
                for k,stat in enumerate(statTags):
                    y, e = local_stat[tag % (1000 * s)][stat]
                    is_best = (y == best_stats[stat][j])
                    if is_best:
                        if LATEX_SHOW_STD:
                            latex.write(" & $\\bm{%.2f}$ ($\pm %.2f$)"%(y, e))
                        else:
                            latex.write(" & $\\bm{%.4f}$" % y)
                    else:
                        if LATEX_SHOW_STD:
                            latex.write(" & $%.2f$ ($\pm %.2f$)"%(y, e))
                        else:
                            latex.write(" & $%.4f$" % y)
            latex.write(" \\\\\n")

    lgd = fig.legend(
        handles, handle_names,
        bbox_to_anchor=(0.65, 0.7), loc='lower center', borderaxespad=0.)
    fig.savefig(os.path.join(output_folder, 'ScreenVsWorld-SSIM.%s'%FILETYPE),
                bbox_inches='tight', bbox_extra_artists=(lgd,))

    # latex footer
    latex.write("\\bottomrule\n")
    latex.write("\\end{tabular}\n")
    latex = latex.getvalue()
    with open(os.path.join(output_folder, "screenVsWorld-SSIM.tex"), 'w') as f:
        f.write(latex)
    print(latex)

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()