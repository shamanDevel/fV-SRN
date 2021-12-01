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

BASE_PATH = 'volnet/results/eval_VolumetricFeatures'
PLOT_LIMIT_TO_SSIM = True # True -> only show SSIM for space reasons, False -> also include LPIPS

configX = [
        ("plume100", "config-files/plume100-v2-dvr.json"),
        ("ejecta70", "config-files/ejecta70-v6-dvr.json"),
        ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json"),
        ("Skull5", "config-files/skull-v5-dvr.json"),
    ]
networkX = []
networkChannelsX = [32, 48, 64]
networkLayersX = [2, 4, 6]
def networkFilename(channels, layers):
    return "l%dx%d"%(channels, layers)
for channels in networkChannelsX:
    for layers in networkLayersX:
        parameters = (channels * (channels+1)) * layers
        networkX.append((networkFilename(channels, layers), channels, layers, parameters))
networkX.sort(key=lambda x: x[3])

fourierX = [("fNeRF", -1)]

volumetricGridSizeX = [4, 8, 16, 32, 64]
volumetricGridChannelX = [4, 8, 16, 32]
def volumetricFilenames(gridSize, gridChannels):
    return "G%dC%d" % (gridSize, gridChannels)
volumetricFeaturesX = [
    (volumetricFilenames(gridSize, gridChannels), ['--volumetric_features_channels', str(gridChannels), '--volumetric_features_resolution', str(gridSize)], {'channels':gridChannels,'size':gridSize}) for
    (gridSize, gridChannels) in itertools.product(volumetricGridSizeX, volumetricGridChannelX)
] + [
(volumetricFilenames(0, 0), ['--volumetric_features_channels', str(0), '--volumetric_features_resolution', str(0)], {'channels':0,'size':0})
]

def main():
    configs = collect_configurations()
    #train(configs)
    statistics_file = eval(configs)
    make_plots(statistics_file)

def collect_configurations():
    cfgs = []
    for config, (i, network), fourier, volumetricFeatures in itertools.product(
            configX, enumerate(networkX), fourierX,volumetricFeaturesX):
        filename = "VolumetricLatentSpace-%04d-%s-%s-%s-%s" % (
            i, config[0], network[0], fourier[0], volumetricFeatures[0])
        cfgs.append((config[1], network[1:], fourier[1], volumetricFeatures[1], filename))
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
        "--lr_step", "50",
        "-i", "200",
        "--logdir", BASE_PATH+'/log',
        "--modeldir", BASE_PATH+'/model',
        "--hdf5dir", BASE_PATH+'/hdf5',
        '--save_frequency', '50'
    ]

    def getNetworkParameters(network):
        channels, layers, params = network
        return ["--layers", ':'.join([str(channels)] * (layers - 1))]

    def getFourierParameters(network, fourier):
        channels, layers, params = network
        std = fourier
        return ['--fouriercount', str((channels - 4) // 2), '--fourierstd', str(std)]

    config, network, fourier, volumetric, filename = cfg

    launcher = [sys.executable, "volnet/train_volnet.py"]
    args = launcher + [config] + \
           common_parameters + \
           getNetworkParameters(network) + \
           getFourierParameters(network, fourier) + \
           volumetric + ['--name', filename]

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

    num_cameras = 2 #64
    width = 512
    height = 512
    STEPSIZE = 1/512
    timer = pyrenderer.GPUTimer()

    #rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = rendering_mode == LoadedModel.EvaluationMode.TENSORCORES_MIXED
    rendering_mode_no_tc = LoadedModel.EvaluationMode.PYTORCH16

    output_stats = []
    device = torch.device('cuda')
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    def compute_stats(ln, mode, reference_images, stepsize, filename_template=None,
                      do_ssim=False, do_lpips=False, render_ref=False):
        timingsX = []
        ssimX = []
        lpipsX = []
        for i in range(num_cameras):
            if render_ref:
                if enable_preintegration:
                    ln.get_image_evaluator().ray_evaluator.convert_to_texture_tf()
                    ln.enable_preintegration(True)
                else:
                    ln.enable_preintegration(False)
                current_image = ln.render_reference(
                    cameras[i], width, height,
                    stepsize_world=stepsize, timer=timer)
            else:
                if ln.is_tensorcores_available() and enable_preintegration:
                    ln.get_image_evaluator().ray_evaluator.convert_to_texture_tf()
                    ln.enable_preintegration(True)
                else:
                    ln.enable_preintegration(False)
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
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            return None, None
        try:
            ln = LoadedModel(filename)
            #if enable_preintegration:
            #    ln.enable_preintegration(True)
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

        reference_images = None
        # collect models
        lns = dict()
        base_ln = None
        for (i, network), fourier, volumetricFeatures in itertools.product(enumerate(networkX), fourierX, volumetricFeaturesX):
            filename = "VolumetricLatentSpace-%04d-%s-%s-%s-%s" % (
                i, config[0], network[0], fourier[0], volumetricFeatures[0])
            ln, name = load_and_save((config[1], network[1:], fourier[1], volumetricFeatures[1], filename))
            lns[(network[0], fourier[0], volumetricFeatures[0])] = (ln, name)
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
        for (i, network), fourier, volumetricFeatures in itertools.product(enumerate(networkX), fourierX, volumetricFeaturesX):
            ln, name = lns[(network[0], fourier[0], volumetricFeatures[0])]
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

        # reference for just the grid
        base_volume_interpolation = base_ln.get_image_evaluator().volume
        base_volume = base_volume_interpolation.volume()
        # Fix for new grid resolution behavior (better match for different grid resolutions)
        # Render reference again
        old_is_new_behavior = base_volume_interpolation.grid_resolution_new_behavior
        base_volume_interpolation.grid_resolution_new_behavior = True
        for i in range(num_cameras):
            reference_images[i] = base_ln.render_reference(cameras[i], width, height)
        # now render the plain grids
        for fourier, volumetricFeatures in itertools.product(fourierX, volumetricFeaturesX):
            name = "VolumetricLatentSpace-xxxx-%s-l0x0-%s-%s" % (
                config[0], fourier[0], volumetricFeatures[0])
            gridParams = volumetricFeatures[2]
            gridChannels = gridParams['channels']
            gridSize = gridParams['size']
            memory = gridChannels * (gridSize**3)
            if memory==0: continue # no-grid network
            target_resolution = int(np.round(np.cbrt(memory)))
            print(f"grid channels {gridChannels}, size {gridSize}^3 -> target size {target_resolution}^3")
            # create a resampled grid for that
            new_volume = base_volume.create_scaled(target_resolution, target_resolution, target_resolution)
            # render with that
            base_volume_interpolation.setSource(new_volume, 0)
            image_folder_screen = os.path.join(image_folder, "%s" % name)
            os.makedirs(image_folder_screen, exist_ok=True)
            time, ssim, lpips = compute_stats(
                base_ln, rendering_mode, reference_images, STEPSIZE,
                os.path.join(image_folder_screen, 'img%03d.png'),
                True, True, render_ref=True)
            local_stats[name] = {
                'time': time,
                'ssim': ssim,
                'lpips': lpips,
            }
        # reset behavior for the upcoming networks
        base_volume_interpolation.grid_resolution_new_behavior = old_is_new_behavior

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

    statNames = ['SSIM $\\uparrow$', 'LPIPS $\\downarrow$']
    statTags = ["ssim", "lpips"]
    def isDegenerated(value, tag):
        if tag == 'ssim': return value < 0.6
        else: return value > 0.5
    statCmaps = [CMAP, CMAP+"_r"]

    fourier = fourierX[0][0]
    # prepare labels
    XlabelTickMinor = []
    XlabelTickMajor = []
    XlabelLabelMinor = []
    XlabelLabelMajor = []
    YlabelTickMinor = []
    YlabelTickMajor = []
    YlabelLabelMinor = []
    YlabelLabelMajor = []
    XlabelTickMajor.append(0.01)
    XlabelLabelMajor.append("$0$")
    XlabelTickMinor.append(0)
    XlabelLabelMinor.append("")
    for ir,r in enumerate(volumetricGridSizeX):
        for ic,c in enumerate(volumetricGridChannelX):
            XlabelTickMinor.append(len(XlabelTickMinor))
            XlabelLabelMinor.append(c)
        XlabelTickMajor.append((ir+0.5) * len(volumetricGridChannelX)+0.5+0.01)
        XlabelLabelMajor.append("$%d^3$"%r)
    YlabelTickMajor.append(0.01)
    YlabelLabelMajor.append("off")
    YlabelTickMinor.append(0)
    YlabelLabelMinor.append("")
    for ir,r in enumerate(networkChannelsX):
        for ic,c in enumerate(networkLayersX):
            YlabelTickMinor.append(len(YlabelTickMinor))
            YlabelLabelMinor.append(c)
        YlabelTickMajor.append((ir+0.5) * len(networkLayersX)+0.5+0.01)
        YlabelLabelMajor.append("%d"%r)

    for config_idx in range(len(configX)):
        local_stat = stats[config_idx]
        num_stats = 1 if PLOT_LIMIT_TO_SSIM else 2
        fig, axs = plt.subplots(num_stats, 1, squeeze=False, figsize=(5, 3*num_stats))
        #axs[row, 0].set_ylabel(configX[row][0])
        for row, (name, tag, cmapName) in enumerate(zip(statNames[:num_stats], statTags[:num_stats], statCmaps[:num_stats])):
            ax = axs[row,0]
            ax.set_title(name)
            data = np.zeros((
                    len(networkChannelsX)*len(networkLayersX)+1,
                    len(volumetricGridSizeX)*len(volumetricGridChannelX)+1))
            allGrids = [(0,0,0)]
            for ri, r in enumerate(volumetricGridSizeX):
                for ci, c in enumerate(volumetricGridChannelX):
                    idxX = ri * len(volumetricGridChannelX) + ci + 1
                    allGrids.append((r,c,idxX))
            for r,c,idxX in allGrids:
                volumetricFeatures = volumetricFilenames(r, c)
                # baseline
                filename = "VolumetricLatentSpace-xxxx-%s-l0x0-%s-%s" % (
                    configX[config_idx][0], fourier, volumetricFeatures)
                if not filename in local_stat:
                    y = np.NaN
                else:
                    y, e = local_stat[filename][tag]
                    #if isDegenerated(y, tag):
                    #    y = np.NaN
                data[0, idxX] = y
                # networks
                for fi, f in enumerate(networkChannelsX):
                    for li, l in enumerate(networkLayersX):
                        idxY = fi*len(networkLayersX) + li
                        network = networkFilename(f, l)
                        networkIdx = list(map(lambda x:x[0], networkX)).index(network)
                        filename = "VolumetricLatentSpace-%04d-%s-%s-%s-%s" % (
                            networkIdx, configX[config_idx][0], network, fourier, volumetricFeatures)
                        if not filename in local_stat:
                            y = np.NaN
                        else:
                            y, e = local_stat[filename][tag]
                            #if isDegenerated(y, tag):
                            #    y = np.NaN
                        data[idxY+1, idxX] = y

                ## legend
                #if row==0 and col==0:
                #    legend_handles.append(h)
                #    legend_names.append(f"{network_channels} channels, {network_layers} layers")
            #ax.set_xticks(X)
            #ticks = ax.set_xticklabels(Xlabel)
            #for tick in ticks: #[-numExtraImportance:]:
            #    tick.set_rotation(45)
            #ax.set_xlabel("Fourier std")

            # make heatmap
            norm = mpl.colors.Normalize(vmin=np.nanmin(data), vmax=np.nanmax(data))
            cmap = mpl.cm.get_cmap(cmapName)
            im = ax.imshow(data, norm=norm, cmap=cmap)
            cbar = ax.figure.colorbar(
                mpl.cm.ScalarMappable(norm=norm,cmap=cmap), ax=ax, shrink=0.7)

            # add numbers to the cells
            #_annotate_heatmap(im, valfmt="{x:.3f}", textcolors=("black", "black"),
            #                  fontsize=3)

            # grid lines
            # have to be added manually because of the manual x ticks
            for x in range(1, 2): #data.shape[1]):
                ax.axvline(x-0.5, 0, 1, color='w', lw=0.3)
            for y in range(1, 2): #data.shape[0]):
                ax.axhline(y-0.5, 0, 1, color='w', lw=0.3)

            ax.set_xticks(XlabelTickMinor, minor=True)
            ax.set_xticklabels(XlabelLabelMinor, minor=True)
            ax.set_xticks(XlabelTickMajor, minor=False)
            xl = ax.set_xticklabels(XlabelLabelMajor, minor=False)
            ax.set_yticks(YlabelTickMinor, minor=True)
            ax.set_yticklabels(YlabelLabelMinor, minor=True)
            ax.set_yticks(YlabelTickMajor, minor=False)
            yl = ax.set_yticklabels(YlabelLabelMajor, minor=False)
            ax.tick_params(axis='x', pad=14)
            ax.tick_params(axis='y', pad=10)
            ax.tick_params(which="major", bottom=False, left=False)
            ax.tick_params(axis='both', which='minor', labelsize=8)
            if row==1: ax.set_xlabel("Grid")
            ax.set_ylabel("Network")
            ax.spines[:].set_visible(False)

        # copy images
        GRIDS_TO_COPY = ["G4C4", "G8C8", "G32C16"]
        NETWORKS_TO_COPY = [("xxxx-%s-l0x0", "l0x0"), ("0000-%s-l32x2", "l32x2")]
        shutil.copyfile(
            os.path.join(output_folder, "images_%s/reference/reference000.png" % (configX[config_idx][0])),
            os.path.join(output_folder, "%s_reference.png" % configX[config_idx][0]))
        for grid, (net, netshort) in itertools.product(GRIDS_TO_COPY, NETWORKS_TO_COPY):
            infile = "VolumetricLatentSpace-" + net%configX[config_idx][0] + "-fNeRF-" + grid
            outfile = netshort + "-" + grid
            shutil.copyfile(
                os.path.join(output_folder, "images_%s/%s/img000.png" % (configX[config_idx][0], infile)),
                os.path.join(output_folder, "%s_%s.png" % (configX[config_idx][0], outfile)))

        fig.subplots_adjust(hspace=-0.05)
        fig.savefig(os.path.join(output_folder, 'VolumetricFeatures-%s.%s'%(configX[config_idx][0], FILETYPE)),
                    bbox_inches='tight')
        plt.close(fig)

    print("Done")
    #plt.show()


if __name__ == '__main__':
    main()