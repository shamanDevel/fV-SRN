"""
Script to evaluate the various network configurations.
They are tested with world-space training and ReLU+Sine activation functions.
"""

import numpy as np
import sys
import os
import io
import subprocess
import itertools
import imageio
import json
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.lines
import matplotlib.collections
from collections import defaultdict

TIMESTEP = 100
CONFIG_FILE = "config-files/plume100-v1-dvr.json"

def main():
    configs = collect_configurations()
    train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    activationX = ["ReLU"] #["ReLU", "Sine", "Snake:2", "SnakeAlt:1"]
    # from collect_possible_layers.py
    networkX = [(32,2), (32,4), (32,10), (32,16), (32,20), (32,23), (48,2), (48,4), (48, 6), (48, 8), (48,10), (64, 2), (64, 4), (64,6), (96, 3), (128, 2)]
    return activationX, networkX

def get_args_and_hdf5_file(activation, network):
    """
    Assembles the command line arguments for training and the filename for the hdf5-file
    with the results
    :param activation: the activation function name
    :param network: the network combination (channels, layers)
    :return: args, filename
    """
    output_name = "run_%s_%s_%s"%(activation.replace(':','-'), network[0], network[1])
    parameters = [
        sys.executable, "volnet/train_volnet.py",
        CONFIG_FILE,
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
        '--fouriercount', str((network[0]-4)//2), '--fourierstd', '1.0',
        "--activation", activation,
        "--layers", ':'.join([str(network[0])]*(network[1]-1)),
        "--logdir", 'volnet/results/eval_network_configs/log',
        "--modeldir", 'volnet/results/eval_network_configs/model',
        "--hdf5dir", 'volnet/results/eval_network_configs/hdf5',
        '--name', output_name,
        '--save_frequency', '50'
    ]
    hdf5_file = 'volnet/results/eval_network_configs/hdf5/' + output_name + ".hdf5"
    return parameters, hdf5_file

def train(configs):
    activationX, networkX = configs
    print("Configurations:", len(activationX)*len(networkX))
    for activation, network in itertools.product(activationX, networkX):
        args, filename = get_args_and_hdf5_file(activation, network)
        if os.path.exists(filename):
            print("Skipping test", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args, check=True)
    print("\n===========================================\nDONE!")

def eval(configs):
    print("Evaluate")
    statistics_file = 'volnet/results/eval_network_configs/stats.json'
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    activationX, networkX = configs
    print("Configurations:", len(activationX) * len(networkX))
    num_cameras = 64
    width = 512
    height = 512
    stepsize = 1 / 256
    timer = pyrenderer.GPUTimer()

    # Rotate around the object, record error and timings
    reference_images = None
    cameras = None
    # stats
    X = [] # x-axis, number of parameters
    labels = [] # x labels (config name)
    Y_performance_ms_pytorch32 = [] # tuples (mean, std)
    Y_performance_ms_pytorch16 = []
    Y_performance_ms_mixed = []
    Y_performance_ms_shared = []
    Y_ssim = []
    Y_lpips = []
    max_steps = None

    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)
    def compute_stats(ln, mode, folder=None, do_ssim=False, do_lpips=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            current_image = ln.render_network(
                cameras[i], width, height, mode,
                stepsize, timer=timer, timestep=TIMESTEP)
            if i>0:
                timingsX.append(timer.elapsed_milliseconds())
            if folder is not None:
                imageio.imwrite(
                    os.path.join(folder, 'frame%03d.png' % i),
                    LoadedModel.convert_image(current_image))
            if do_ssim:
                ssimX.append(ssim_loss(current_image, reference_images[i]).item())
            if do_lpips:
                lpipsX.append(lpips_loss(current_image, reference_images[i]).item())
        return \
            (np.mean(timingsX), np.std(timingsX)), \
            (np.mean(ssimX), np.std(ssimX)) if do_ssim else (np.NaN, np.NaN), \
            (np.mean(lpipsX), np.std(lpipsX)) if do_lpips else (np.NaN, np.NaN), \

    for activation, network in itertools.product(activationX, networkX):
        args, filename = get_args_and_hdf5_file(activation, network)

        #if filename!='volnet/results/eval_network_configs/hdf5/run_Sine_32_10.hdf5':
        #    continue

        if not os.path.exists(filename):
            print("Missing test", filename)
        image_folder = filename.replace('/hdf5/', '/images/').replace('.hdf5', '')
        os.makedirs(image_folder, exist_ok=True)
        reference_folder = os.path.join(image_folder, '../reference')
        os.makedirs(reference_folder, exist_ok=True)
        # load model
        ln = LoadedModel(filename, force_config_file=CONFIG_FILE)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        ln.set_alpha_early_out(False)
        if not ln.is_tensorcores_available():
            print("\n=====================================", filename, "too big for TC. Skipping")
            continue
        # render reference, if not done yet
        if reference_images is None:
            print("\n===================================== Render reference")
            cameras = ln.get_rotation_cameras(num_cameras)
            if max_steps is None:
                max_steps = max([ln.get_max_steps(c, width, height, stepsize) for c in cameras])
            reference_images = [None] * num_cameras
            for i in range(num_cameras):
                reference_images[i] = ln.render_reference(cameras[i], width, height, timestep=TIMESTEP)
                imageio.imwrite(
                    os.path.join(reference_folder, 'frame%03d.png'%i),
                    LoadedModel.convert_image(reference_images[i]))
        # render this network
        print("\n===================================== Render", filename)
        print("Warps shared:", ln.warps_shared(), ", mixed:", ln.warps_mixed())
        X.append(ln.num_parameters())
        labels.append(("%s"%activation, "%d^%d" % (network[0], network[1])))

        time, ssim, lpips = compute_stats(
            ln, LoadedModel.EvaluationMode.PYTORCH32, image_folder, True, True)
        Y_ssim.append(ssim)
        Y_lpips.append(lpips)
        Y_performance_ms_pytorch32.append(time)

        time, _, _ = compute_stats(
            ln, LoadedModel.EvaluationMode.PYTORCH16)
        Y_performance_ms_pytorch16.append(time)

        if ln.is_tensorcores_available():
            time, _, _ = compute_stats(
                ln, LoadedModel.EvaluationMode.TENSORCORES_MIXED)
            Y_performance_ms_mixed.append(time)
            time, _, _ = compute_stats(
                ln, LoadedModel.EvaluationMode.TENSORCORES_SHARED)
            Y_performance_ms_shared.append(time)
        else:
            Y_performance_ms_mixed.append((np.NaN, np.NaN))
            Y_performance_ms_shared.append((np.NaN, np.NaN))


    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file, "w") as f:
        json.dump({
            'X': X,
            'labels': labels,
            'Y_performance_ms_pytorch32': Y_performance_ms_pytorch32,
            'Y_performance_ms_pytorch16': Y_performance_ms_pytorch16,
            'Y_performance_ms_shared': Y_performance_ms_shared,
            'Y_performance_ms_mixed': Y_performance_ms_mixed,
            'Y_ssim': Y_ssim,
            'Y_lpips': Y_lpips,
            'max_steps': max_steps,
        }, f)
    return statistics_file

def make_plots(statistics_file):
    print("\n===================================== Make Plots")
    with open(statistics_file, "r") as f:
        stats = json.load(f)
    output_folder = os.path.split(statistics_file)[0]
    FILETYPE = "eps"

    X = stats['X']
    labels = stats['labels']
    activations = [l[0] for l in labels]
    networks = [l[1] for l in labels]
    networks2 = map(lambda x: (int(x[0]), int(x[1])), map(lambda x: x.split('^'), set(networks)))
    networks2 = list(sorted(networks2))
    PERFORMANCE_KEY = "ReLU"

    largest_layers_per_channel = defaultdict(lambda: 0)
    for channels, layers in networks2:
        if largest_layers_per_channel[channels] < layers:
            largest_layers_per_channel[channels] = layers

    # plot 1: SSIM + LPIPS
    # build labels
    x = 1
    network2x = dict()
    x_pos = []
    x_labels = []
    x_keys = []
    for channels,layers in networks2:
        x_pos.append(x)
        x_labels.append("%d\n%d"%(channels, layers))
        x_keys.append("%d^%d" % (channels, layers))
        network2x["%d^%d"%(channels, layers)] = x
        x += 1

    x_filtered = 1
    network2x_filtered = dict()
    x_pos_filtered = []
    x_labels_filtered = []
    x_keys_filtered = []
    for channels, layers in networks2:
        if largest_layers_per_channel[channels] != layers: continue  # only show largest per channels
        x_pos_filtered.append(x_filtered)
        x_labels_filtered.append("%d\n%d" % (channels, layers))
        x_keys_filtered.append("%d^%d" % (channels, layers))
        network2x_filtered["%d^%d" % (channels, layers)] = x_filtered
        x_filtered += 1

    fig, axs = plt.subplots(1, 2)
    activation2handle = {}
    for ax, stat_name, name in zip(axs, ['Y_ssim', 'Y_lpips'], ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']):
        stat = stats[stat_name]
        #for n, a, (y_mean, y_std) in zip(networks, activations, stat):
        #    h = ax.errorbar([network2x[n]], [y_mean], [y_std], marker=MARKERS[a], color='black', capsize=2)
        #    activation2handle[a] = h
        unique_activations = list(sorted(set(activations)))
        x_offsets = np.linspace(-0.02, +0.02, len(unique_activations), True)
        for ai,activation in enumerate(unique_activations):
            y = []
            err = []
            for i, key in enumerate(x_keys_filtered):
                j = labels.index([activation, key])
                y.append(stat[j][0])
                err.append(stat[j][1])
            # fix outlier
            median = np.median(y)
            for i in range(len(y)):
                outlier = y[i] < 0.7 * median if stat_name == 'Y_ssim' else y[i] > 3*median
                if outlier:
                    y[i] = np.NaN
                    err[i] = np.NaN
            h = ax.errorbar(
                np.array(x_pos_filtered)+x_offsets[ai], y, err,
                marker='o', capsize=2, linestyle='None',#,alpha=.55,
                label=activation.split(':')[0])
            activation2handle[activation] = h
        ax.set_title(name)
        ax.set_xlim(left=0.8, right=x_filtered - 0.8)
        ax.grid(True, linestyle='-.')
        if stat_name == 'Y_ssim':
            ax.set_xticks([1-1e-3]+x_pos_filtered)
            tx = ax.set_xticklabels(['channels:   \nlayers:   ']+x_labels_filtered)
            tx[0].set_ha('right')
            #tx[0].set_x(5)
            pass
        else:
            ax.set_xticks(x_pos_filtered)
            ax.set_xticklabels(x_labels_filtered)
        #fmt = matplotlib.ticker.ScalarFormatter()
        #fmt.set_powerlimits((-3, 3))
        #ax.yaxis.set_major_formatter(fmt)
        #for x, a, (y_mean, y_std) in zip(X, activations, stat):
        #    ax.errorbar([x], [y_mean], [y_std], marker=MARKERS[a], color='black', capsize=2)
        #ax.set_title(name)
        #ax._set_xlabel('# parameters')
        #activation2handle = list(activation2handle.items())
        if stat_name == 'Y_lpips':
            lgd = ax.legend(#[a[1] for a in activation2handle], [a[0] for a in activation2handle],
               bbox_to_anchor=(0.75, 0.8), loc='center right', borderaxespad=0.)
    fig.subplots_adjust(wspace=0.25)
    fig.savefig(os.path.join(output_folder, 'NetworkConfigs-SSIM-LPIPS.%s'%FILETYPE),
                bbox_inches='tight', bbox_extra_artists=(lgd,))

    # plot 2: frame time
    fig = plt.figure()
    ax = plt.gca()
    lines = []
    for stat_id, stat_name in zip(
            ['Y_performance_ms_pytorch32', 'Y_performance_ms_pytorch16', 'Y_performance_ms_shared'],#, 'Y_performance_ms_mixed'],
            ['PyTorch 32', 'PyTorch 16', 'Ours'] #'TC shared', 'TC mixed']
            ):
        stat = stats[stat_id]
        y = []
        err = []
        for i,key in enumerate(x_keys):
            j = labels.index([PERFORMANCE_KEY, key])
            y.append(stat[j][0]/1000)
            err.append(stat[j][1]/1000)
        lines.append(ax.errorbar(x_pos, y, err, marker='o', capsize=2, linestyle='None', label=stat_name))
        #lines.append(ax.plot(x_pos, y, marker='o', label=stat_name))
    # separators
    prev_key = x_keys[0]
    for x,key in zip(x_pos, x_keys):
        if key[:3] != prev_key[:3]:
            l = ax.axvline(x-0.5, -0.1, 1-0.001, color='k', lw=0.4)
            l.set_clip_on(False)
            prev_key = key
    # limits
    ax.set_xlim(left=0.8, right=x_pos[-1]+0.2)
    ax.grid(True, linestyle='-.')
    ax.set_xticks([1-1e-3] + x_pos)
    ax.set_ylabel('Time in seconds')
    tx = ax.set_xticklabels(['channels:   \nlayers:   '] + x_labels)
    tx[0].set_ha('right')
    lgd = fig.legend(bbox_to_anchor=(0.85, 0.7), loc='center right', borderaxespad=0.)
    def disable_lines(tuple_or_line):
        if isinstance(tuple_or_line, matplotlib.lines.Line2D):
            tuple_or_line.set_visible(False)
        if isinstance(tuple_or_line, matplotlib.collections.LineCollection):
            tuple_or_line.set_visible(False)
        else:
            try:
                for l in tuple_or_line:
                    disable_lines(l)
            except TypeError:
                pass # not iterable
    for i in range(3):
        if i == 1: disable_lines(lines[2])
        if i == 2: disable_lines(lines[1])
        fig.savefig(os.path.join(output_folder, 'NetworkConfigs-Performance-%d.%s'%(i,FILETYPE)),
                    bbox_inches='tight', bbox_extra_artists=(lgd,))

    print("Done")
    plt.show()

    # Table with speedup
    str = io.StringIO()
    print("Network  | PyTorch32 (ms) | PyTorch16 (ms) | TensorCores (ms) | Speedup Torch32-TC | Speedup Torch16-TC |", file=str)
    for i, key in enumerate(x_keys):
        j = labels.index([PERFORMANCE_KEY, key])
        network = key
        pytorch32_ms = stats['Y_performance_ms_pytorch32'][j][0]
        pytorch16_ms = stats['Y_performance_ms_pytorch16'][j][0]
        TC_ms = stats['Y_performance_ms_shared'][j][0]
        speedup_torch32 = pytorch32_ms / TC_ms
        speedup_torch16 = pytorch16_ms / TC_ms
        print(f" {key:<7} |{pytorch32_ms:^16.5}|{pytorch16_ms:^16.5}|{TC_ms:^18.5}|{speedup_torch32:^21.5f}|{speedup_torch16:^21.5f}|", file=str)
    print("Max steps:", stats['max_steps'], file=str)
    str = str.getvalue()
    print(str)
    with open(os.path.join(output_folder, "NetworkConfigs-Performance_Stats.txt"), "w") as f:
        f.write(str)

if __name__ == '__main__':
    main()