import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import sys
import os
import subprocess
import random
import imageio
import json
import itertools
import shutil
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
import matplotlib.patches as patches
import seaborn as sns
from collections import defaultdict
from typing import NamedTuple, Tuple, List, Optional, Any, Union

import common.utils as utils
import pyrenderer
from volnet.inference import LoadedModel

BASE_PATH = 'volnet/results/eval_CurvatureNetworks2'

class Config(NamedTuple):
    name: str
    settings_train: str
    settings_eval: str
    rendering_mode: LoadedModel.EvaluationMode
    args: List[str]

configX = [
    Config(
        name = "PuffyCube",
        settings_train = "config-files/implicit-PuffyCube-v2-dvr.json",
        settings_eval = "config-files/implicit-PuffyCube-v2-GaussianCurvature.json",
        rendering_mode = LoadedModel.EvaluationMode.PYTORCH32,
        args = [
            "--train:mode", "world",
            "--train:samples", "512**3",
            "--train:batchsize", "64*64*128",
            "--val:copy_and_split",
            "--rebuild_dataset", "51", "--rebuild_importance", "0.01",
            "--rebuild_force_color",
            "--layers", "32:32:32", "--outputmode", "densitycurvature:direct",
            "--activation", "SnakeAlt:1",
            "--fouriercount", "14", "--fourierstd", "-1",
            "--volumetric_features_channels", "16", "--volumetric_features_resolution", "32",
            "--lossmode", "densitycurvature", "-l1", "1",
            "--gradient_l1", "0", "--gradient_l2", "1",
            "--gradient_weighting", "0.006692850924284843", "--curvature_l2", "0.05",
            "-lr", "0.01", "--lr_step", "100", "-i", "300",
            "--seed", "44"]
    ),
    Config(
        name = "Skull",
        settings_train = "config-files/skull-v9-dvr-curvature-noSmoothing.json",
        settings_eval = "config-files/skull-v9-iso-curvature2.json",
        rendering_mode=LoadedModel.EvaluationMode.TENSORCORES_MIXED,
        args = [
            "--train:mode", "world",
            "--train:samples", "512**2*1024",
            "--train:batchsize", "64*64*128",
            "--rebuild_dataset", "51",
            "--rebuild_importance", "0.1", "--rebuild_force_color",
            "--clamp_values", "500", "--val:copy_and_split",
            "--layers", "32:32:32",
            "--outputmode", "densitycurvature:direct",
            "--activation", "SnakeAlt:1",
            "--fouriercount", "14",
            "--fourierstd", "-1",
            "--volumetric_features_channels", "16",
            "--volumetric_features_resolution", "64",  # 32
            "--lossmode", "densitycurvature",
            "-l1", "1", "--gradient_l1", "0", "--gradient_l2", "1",
            "--gradient_weighting", "0.01798620996209155",
            "--curvature_l2", "0.00012339457598625758",
            "--weight_gradient_curvature_by_opacity",
            "-lr", "0.002", "--lr_step", "100",
            "-i", "200",
            "--save_frequency", "20", "--seed", "43"
        ]
    )
]

def main():
    for config in configX:
        print("\n==========================================")
        print(config.name)
        print("==========================================")

        hdf_file = train(config)
        eval(config, hdf_file)


def train(config: Config):
    args = [
        sys.executable,
        "volnet/train_volnet.py",
        config.settings_train,
    ] + config.args + [
        "--logdir", BASE_PATH + '/log',
        "--modeldir", BASE_PATH + '/model',
        "--hdf5dir", BASE_PATH + '/hdf5',
        "--name", config.name
    ]
    hdf_file = os.path.join(BASE_PATH, 'hdf5', config.name+".hdf5")
    if os.path.exists(os.path.join(BASE_PATH, 'hdf5', config.name+".hdf5")):
        print("Skipping", config.name)
    else:
        print("\n=====================================\nRun", config.name)
        subprocess.run(args, check=True)
    return hdf_file

def eval(config: Config, hdf_file: str):

    width = 1920
    height = 1080
    stepsize = 1 / 128

    ln = LoadedModel(hdf_file, force_config_file=config.settings_eval)
    ln.save_compiled_network(hdf_file.replace('.hdf5', '.volnet'))

    image_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(image_folder, exist_ok=True)

    # render reference
    camera = ln.get_default_camera()
    reference_image = ln.render_reference(
        camera, width, height, timer=None, stepsize_world=stepsize,
        channel=pyrenderer.IImageEvaluator.Color)
    imageio.imwrite(
        os.path.join(image_folder, '%s-color-reference.png' % config.name),
        LoadedModel.convert_image(reference_image))

    # render network
    current_image = ln.render_network(
        camera, width, height, config.rendering_mode, stepsize,
        channel=pyrenderer.IImageEvaluator.Color)
    imageio.imwrite(
        os.path.join(image_folder, '%s-color-network.png' % config.name),
        LoadedModel.convert_image(current_image))

if __name__ == '__main__':
    main()