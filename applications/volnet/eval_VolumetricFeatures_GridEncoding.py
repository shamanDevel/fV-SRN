"""
REQUIRES eval_VolumetricFeatures.py to be run beforehand!
Evaluates the impact of the volume encoding on the quality
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

configX = [
        ("plume100", "config-files/plume100-v2-dvr.json"),
        ("ejecta70", "config-files/ejecta70-v6-dvr.json"),
        ("RM60", "config-files/RichtmyerMeshkov-t60-v1-dvr.json"),
    ]

SELECTED_NETWORK_CHANNELS = 32
SELECTED_NETWORK_LAYERS = 4
SELECTED_FOURIER = "fNeRF"
SELECTED_GRID_SIZE = 32
SELECTED_GRID_CHANNEL = 16

def _networkFilename(channels, layers):
    return "l%dx%d"%(channels, layers)

def _volumetricFilenames(gridSize, gridChannels):
    return "G%dC%d" % (gridSize, gridChannels)

networkX = []
networkChannelsX = [32, 48, 64]
networkLayersX = [2, 4, 6]
for channels in networkChannelsX:
    for layers in networkLayersX:
        parameters = (channels * (channels+1)) * layers
        networkX.append((_networkFilename(channels, layers), channels, layers, parameters))
networkX.sort(key=lambda x: x[3])

def getFilename(config):
    networkIdx = None
    fn = _networkFilename(SELECTED_NETWORK_CHANNELS, SELECTED_NETWORK_LAYERS)
    gn = _volumetricFilenames(SELECTED_GRID_SIZE, SELECTED_GRID_CHANNEL)
    for i,net in enumerate(networkX):
        if net[0] == fn:
            networkIdx = i
            break
    return "VolumetricLatentSpace-%04d-%s-%s-%s-%s" % (
        networkIdx, config[0], fn, SELECTED_FOURIER, gn)

def main():
    statistics_file = eval_and_plot()
    make_plots(statistics_file)

def eval_and_plot():
    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats_GridEncoding.json')
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        #return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel
    from losses.lossbuilder import LossBuilder

    num_cameras = 4#64
    width = 1024#512
    height = 1024#512
    STEPSIZE = 0.005 #1/512
    timer = pyrenderer.GPUTimer()
    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
    #rendering_mode = LoadedModel.EvaluationMode.PYTORCH16
    enable_preintegration = rendering_mode==LoadedModel.EvaluationMode.TENSORCORES_MIXED

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
                current_image = ln.render_reference(
                    cameras[i], width, height,
                    stepsize_world=stepsize, timer=timer)
            else:
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
    def load_and_save(filename, encoding, encoding_name):
        filename2 = os.path.abspath(os.path.join(BASE_PATH, "hdf5", filename+".hdf5"))
        if not os.path.exists(filename2):
            print("File not found:", filename2, file=sys.stderr)
            return None
        try:
            ln = LoadedModel(filename2, grid_encoding=encoding)
            if enable_preintegration:
                ln.get_image_evaluator().ray_evaluator.convert_to_texture_tf()
                ln.enable_preintegration(True)
            filename3 = os.path.abspath(os.path.join(BASE_PATH, "hdf5", filename + "_" + encoding_name + ".volnet"))
            ln.save_compiled_network(filename3)
            return ln
        except Exception as e:
            print("Unable to load '%s':"%filename, e)
            return None

    for cfg_index, config in enumerate(configX):
        image_folder = os.path.join(BASE_PATH, "imagesGridEncoding_"+config[0])
        local_stats = {
            'cfg_index': cfg_index,
            'cfg': config[1]}

        reference_images = None
        # collect models
        filename = getFilename(config)
        lns = {
            'float': load_and_save(
                filename, pyrenderer.SceneNetwork.LatentGrid.Float, "float"),
            'linear': load_and_save(
                filename, pyrenderer.SceneNetwork.LatentGrid.ByteLinear, "linear"),
            'gaussian': load_and_save(
                filename, pyrenderer.SceneNetwork.LatentGrid.ByteGaussian, "gaussian")
        }
        base_ln = lns['float']

        # test gaussian
        import scipy.stats
        grid = base_ln.get_network_pytorch()[0]._volumetric_latent_space
        print("Grid shape:", grid.shape)
        gridc = grid[:,0,...] # select channel 0
        grid_min = torch.min(gridc).item()
        grid_max = torch.max(gridc).item()
        grid_mean = torch.mean(gridc).item()
        grid_std = torch.std(gridc).item()
        print(f"Grid min={grid_min}, max={grid_max}, mean={grid_mean}, std={grid_std}")
        #grid_min = -1
        #grid_max = +1
        grid_flat = gridc.detach().cpu().numpy().flatten()
        Xgaussian = np.linspace(grid_min, grid_max, 200)
        Ygaussian = scipy.stats.norm.pdf(Xgaussian, loc=grid_mean, scale=grid_std)
        fig = plt.figure()
        ax = plt.gca()
        ax.hist(grid_flat, range=(grid_min,grid_max), bins=100, density=True)
        ax.plot(Xgaussian, Ygaussian)
        ax.set_xlabel("Value of Volumetric Latent Space, channel 0")
        ax.set_ylabel("Probability")
        #ax.set_yscale("log")
        fig.savefig(os.path.join(BASE_PATH, 'VolumetricFeatures_GridEncoding_Distribution-%s.eps' % (config[0])),
                    bbox_inches='tight')
        plt.close(fig)

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
        for key,ln in lns.items():
            print("Render", key)
            image_folder_screen = os.path.join(image_folder, "%s" % key)
            os.makedirs(image_folder_screen, exist_ok=True)
            time, ssim, lpips = compute_stats(
                ln, rendering_mode, reference_images, STEPSIZE,
                os.path.join(image_folder_screen, 'img%03d.png'),
                True, True)
            local_stats[key] = {
                'encoding_error': ln.get_grid_encoding_error(),
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

    print("Done")
    #plt.show()
    # Print error and degradation
    str = io.StringIO()
    for cfg_index, config in enumerate(configX):
        print("\n", config[0], file=str)
        local_stats = stats[cfg_index]
        print("Encoding  |   Error   |   SSIM    |   LPIPS   | Time (ms) |", file=str)
        reference_key = 'float'
        keys = ['float', 'linear', 'gaussian']
        ref_ssim = local_stats[reference_key]['ssim'][0]
        ref_lpips = local_stats[reference_key]['lpips'][0]
        for key in keys:
            error = local_stats[key]['encoding_error']
            ssim = local_stats[key]['ssim'][0]
            lpips = local_stats[key]['lpips'][0]
            time = local_stats[key]['time'][0]
            print(f" {key:<8} |{error:^11.5}|{ssim:^11.5}|{lpips:^11.5}|{time:^11.2f}|", file=str)
    str = str.getvalue()
    print(str)
    with open(os.path.join(BASE_PATH, "VolumetricFeatures_GridEncoding_Stats.txt"), "w") as f:
        f.write(str)


if __name__ == '__main__':
    main()