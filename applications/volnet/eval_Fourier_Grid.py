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
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
from collections import defaultdict

BEST_ACTIVATION = "SnakeAlt:1"
BEST_NETWORK = (32,4)
GRID_RESOLUTION = 32
GRID_CHANNELS = 16

BASE_PATH = 'volnet/results/eval_FourierGrid'

configX = [
    ("plume100", "config-files/plume100-v2-dvr.json"),
    ("ejecta70", "config-files/ejecta70-v6-dvr.json"),
    ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json"),
]
HUMAN_NAMES = {
    "plume100": "ScalarFlow",
    "ejecta70": "Ejecta",
    "RM60": "RM"
}
networkX = [
    ("l%dx%d"%BEST_NETWORK, BEST_NETWORK[0], BEST_NETWORK[1]),
]
fourierStdX = [
    ("f%03d"%int(f*10), f) for f in [0.1, 0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0]
] + [("fNerf", -1)]
fourierChannelsX = [((channels - 4) // 2) for channels in [16, 32, 48]]
def _getFourierName(std, c):
    return "%sX%d"%(std[0], c)
fourierX = [
    (_getFourierName(std, c),
     ['--fouriercount', str(c), '--fourierstd', str(std[1])])
    for std,c in itertools.product(fourierStdX, fourierChannelsX)
]
fourierX.append(("fOff", []))
#fourierX = [("fNeRF", -1)]

def main():
    configs = collect_configurations()
    #train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    for config, network, fourier in itertools.product(configX, networkX, fourierX):
        filename = "fourier-world-%s-%s-%s" % (
            config[0], network[0], fourier[0])
        cfgs.append((config[1], network[1:], fourier[1], filename))
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
        "--volumetric_features_resolution", str(GRID_RESOLUTION),
        "--volumetric_features_channels", str(GRID_CHANNELS),
        "-l1", "1",
        "--lr_step", "50",
        "-i", "200",
        "--logdir", BASE_PATH+'/log',
        "--modeldir", BASE_PATH+'/model',
        "--hdf5dir", BASE_PATH+'/hdf5',
    ]
    def getNetworkParameters(network):
        channels, layers = network
        return ["--layers", ':'.join([str(channels)] * (layers - 1))]

    config, network, fourier, filename = cfg

    launcher = [sys.executable, "volnet/train_volnet.py"]
    args = launcher + [config] + \
           common_parameters + \
           getNetworkParameters(network) + \
           fourier + \
           ['--name', filename]

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
    #rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
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
        ln.enable_preintegration(enable_preintegration)
        ln.save_compiled_network(filename.replace('.hdf5', '.volnet'))
        return ln, output_name

    """
    for config, network, fourier in itertools.product(configX, networkX, fourierX):
        filename = "fourier-world-%s-%s-%s" % (
            config[0], network[0], fourier[0])
        cfgs.append((config[1], network[1:], fourier[1], filename))
    """

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "images_"+config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        reference_images = None
        # collect models
        lns = dict()
        base_ln = None
        for network, fourier in itertools.product(networkX, fourierX):
            filename = "fourier-world-%s-%s-%s" % (
                config[0], network[0], fourier[0])
            ln, name = load_and_save((config[1], network[1:], fourier[1], filename))
            lns[(network[0], fourier[0])] = (ln, name)
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
        for network, fourier in itertools.product(networkX, fourierX):
            ln, name = lns[(network[0], fourier[0])]
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



def _heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()
    if not cbar_kw:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    #ax.tick_params(top=True, bottom=False,
    #               labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    #plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
    #         rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def _annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts



def make_plots(statistics_file):
    print("\n===================================== Make Plots")
    with open(statistics_file, "r") as f:
        stats = json.load(f)
    output_folder = os.path.split(statistics_file)[0]
    FILETYPE = "eps"

    CMAP = "viridis"
    NETWORK_KEY = networkX[0][0]

    # only show SSIM for space reasons

    numRows = len(configX)
    statNames = ['SSIM $\\uparrow$']#, 'LPIPS $\\downarrow$']
    statTags = ["ssim"]#, "lpips"]
    statCmaps = [CMAP]#, CMAP + "_r"]
    numCols = len(statTags)

    # X: fourier std
    # Y: channels
    def _format(v):
        if v<0: return "NeRF"
        if v<1: return "%.1f"%v
        return "%d"%int(v)
    Xlabels = [_format(v[1]) for v in fourierStdX] + ["Off"]
    Ylabels = ["%d"%v for v in fourierChannelsX]

    fig, axs = plt.subplots(numRows, numCols, squeeze=False,
                            sharex=True, sharey=True, figsize=(5, 1.5 * numRows))
    legend_handles = []
    legend_names = []
    norm = None
    cmap = None
    for row in range(numRows):
        local_stat = None
        for stat in stats:
            if stat['cfg'] == configX[row][1]:
                local_stat = stat
                break
        assert local_stat != None
        #axs[row, 0].set_ylabel(configX[row][0])
        for col, (name, tag, cmap) in enumerate(zip(statNames, statTags, statCmaps)):
            ax = axs[row,col]
            #if row==0:
            #    ax.set_title(name)
            ax.set_title(HUMAN_NAMES[configX[row][0]])

            data = np.zeros((len(Ylabels), len(Xlabels)))
            for x,std in enumerate(fourierStdX):
                for y,c in enumerate(fourierChannelsX):
                    filename = "fourier-world-%s-%s-%s" % (
                        configX[row][0], NETWORK_KEY, _getFourierName(std,c))
                    data[y,x] = local_stat[filename][tag][0]
            for y,c in enumerate(fourierChannelsX):
                filename = "fourier-world-%s-%s-%s" % (
                    configX[row][0], NETWORK_KEY, "fOff")
                data[y,len(fourierStdX)] = local_stat[filename][tag][0]

            norm = mpl.colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            cmap = mpl.cm.get_cmap(cmap)
            im = ax.imshow(data, norm=norm, cmap=cmap)
            #cbar = ax.figure.colorbar(
            #    mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.7)
            _annotate_heatmap(im, valfmt="{x:.3f}", textcolors=("darkgray", "black"),
                              fontsize=6)

            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_xticklabels(Xlabels)
            ax.set_yticklabels(Ylabels)
            if row == numRows-1:
                ax.set_xlabel("Fourier Std $\sigma$")
            ax.set_ylabel("Channels")

        ## determine and copy best and worst images
        #tag = "lpips"
        #worst_lpips = 0
        #worst_filename = None
        #best_lpips = 1
        #best_filename = None
        #for network_label, network_channels, network_layers in networkX:
        #      for i, (fn, f) in enumerate(fourierX[:-1]):
        #        filename = "fourier-world-%s-%s-%s" % (
        #            configX[row][0], network_label, fn)
        #        y, _ = local_stat[filename][tag]
        #        if y < best_lpips:
        #            best_lpips = y
        #            best_filename = filename
        #        if y > worst_lpips:
        #            worst_lpips = y
        #            worst_filename = filename
#
        #shutil.copyfile(
        #    os.path.join(output_folder, "images_%s/reference/reference000.png" % (configX[row][0])),
        #    os.path.join(output_folder, "%s_reference.png" % configX[row][0]))
        #shutil.copyfile(
        #    os.path.join(output_folder, "images_%s/%s/img000.png"%(configX[row][0], best_filename)),
        #    os.path.join(output_folder, "%s_best.png"%configX[row][0]))
        #shutil.copyfile(
        #    os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[row][0], worst_filename)),
        #    os.path.join(output_folder, "%s_worst.png" % configX[row][0]))

    #colorbar
    fig.subplots_adjust(right=0.8,hspace=0.3)
    cb_ax = fig.add_axes([0.83, 0.1, 0.05, 0.8])
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cb_ax)

    #lgd = fig.legend(
    #    legend_handles, legend_names,
    #    #bbox_to_anchor=(0.75, 0.7), loc='lower center', borderaxespad=0.
    #    loc='upper center', bbox_to_anchor=(0.5, 0.05),
    #    ncol=len(legend_handles))
    fig.savefig(os.path.join(output_folder, 'FourierGrid-SSIM.%s'%FILETYPE),
                bbox_inches='tight')#, bbox_extra_artists=(lgd,))

    print("Done")
    plt.show()


if __name__ == '__main__':
    main()