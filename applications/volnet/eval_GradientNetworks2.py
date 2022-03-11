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
from volnet.network_gradients import NetworkGradientTransformer
from losses.lossbuilder import LossBuilder
from volnet.sampling import PlasticSampler

BASE_PATH = 'volnet/results/eval_GradientNetworks2'

BEST_GRID_RESOLUTION = 32
BEST_GRID_CHANNELS = 16
BEST_NETWORK_LAYERS = 4
BEST_NETWORK_CHANNELS = 32
BEST_ACTIVATION = "SnakeAlt:1"
BEST_FOURIER_STD = -1 # NERF
BEST_FOURIER_COUNT = 14 # to fit within 32 channels

DEFAULT_NUM_SAMPLES = "512**3"
DEFAULT_NUM_EPOCHS = 100
DEFAULT_STEPSIZE = 1 / 512

SAMPLER_IMPORTANCE = 0.3
REBUILD_DATASET_EPOCHS = 35
REBUILD_DATASET_SAMPLES = 32
REBUILD_DATASET_IMPORTANCES = [0.01, 0.1, 0.2]
NETWORK_OUTPUT_MODES = ["densitygrad:direct", "densitygrad:cubic"]
LOSSES_DENSITY = ['l1']
LOSSES_GRADIENT = ['l1', 'l2']
SEEDS = [42, 43, 44]

GRADIENT_WEIGHT_RANGE_MAX = 0
GRADIENT_WEIGHT_RANGE_MIN = -9 #-6
GRADIENT_WEIGHT_SCALE = 0.5

EVAL_WORLD_NUM_POINTS = 256**3 #512**3
EVAL_SCREEN_SIZE = 1024
EVAL_LENGTH_THRESHOLDS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
EVAL_LENGTH_THRESHOLDS_IDX_PLOT = 1

class PlotHighlight(NamedTuple):
    gradient_weight: int
    output_mode:str
    loss_gradient: str
    dataset_importance_key: str
    color: str

EVAL_HIGHLIGHTS = [
    PlotHighlight( # good config, used for all other datasets
        gradient_weight=-6,
        output_mode='densitygrad:direct',
        loss_gradient='l2',
        dataset_importance_key='$p$=0.1, $c$=False',
        color='green'
    ),
    PlotHighlight( # bad config
        gradient_weight=-2,
        output_mode='densitygrad:cubic',
        loss_gradient='l2',
        dataset_importance_key='$p$=0.01, $c$=False',
        color='blue'
    )
]

class Config(NamedTuple):
    name: str
    settings: str
    grid_size: int
    overwrite_layers: Optional[int] = None
    overwrite_samples: Optional[str] = None
    overwrite_epochs: Optional[int] = None

class Run(NamedTuple):
    config: Config
    filename: str
    # data
    do_rebuild: bool
    rebuild_importance: int
    rebuild_use_color: bool
    seed: int
    # network
    network_output_mode: str
    # loss
    loss_density: str
    loss_gradient: str
    gradient_weight_index: int
    # command line
    args: List[str]

configX = [
    Config(
        name = "Skull",
        settings = "config-files/skull-v8-dvr-shaded.json",
        grid_size = 256
    ),
    Config(
        name = "RM512",
        settings = "config-files/RM512-v2-shaded.json",
        grid_size = 512
    )
]

def main():
    for config in configX:
        print("\n==========================================")
        print(config.name)
        print("==========================================")

        runs = assemble_runs(config)
        #create_scripts(config, runs)

        # train(config, runs)
        statistics_file = eval(config, runs)
        make_plots1(config, runs, statistics_file)
        make_plots2(config, runs, statistics_file)

def _gradient_weight(index: int):
    """
    Converts from the weight index in [GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX]
    to the actual gradient weight in [0,1]
    """
    return np.tanh(GRADIENT_WEIGHT_SCALE*index)*0.5 + 0.5

def assemble_runs(config: Config):
    runs: List[Run] = []

    best_network_layers = config.overwrite_layers or BEST_NETWORK_LAYERS
    training_samples = config.overwrite_samples or DEFAULT_NUM_SAMPLES
    epochs = config.overwrite_epochs or DEFAULT_NUM_EPOCHS

    for weight_index in range(GRADIENT_WEIGHT_RANGE_MIN, GRADIENT_WEIGHT_RANGE_MAX + 1):
        for importance,color in [(0, False)] + [(i, False) for i in REBUILD_DATASET_IMPORTANCES] + [(i, True) for i in REBUILD_DATASET_IMPORTANCES]:
            for om in NETWORK_OUTPUT_MODES:
                for ld in LOSSES_DENSITY:
                    for lg in LOSSES_GRADIENT:
                        for seed in SEEDS:
                            filename = f"{config.name}_W{-weight_index}_I{int(100*importance):d}-C{int(color)}_{om.split(':')[1]}_D{ld}-G{lg}_Seed{seed}"
                            args = [
                                sys.executable, "volnet/train_volnet.py",
                                config.settings,
                                "--train:mode", "world",
                                "--train:samples", training_samples,
                                "--train:batchsize", "64*64*128",
                                '--rebuild_dataset', str(REBUILD_DATASET_EPOCHS) if importance>0 else "-1",
                                '--rebuild_importance', str(importance),
                            ] + (['--rebuild_force_color'] if color else []) + [
                                "--val:copy_and_split",
                                "--layers", ':'.join([str(BEST_NETWORK_CHANNELS)] * (best_network_layers - 1)),
                                # -1 because last layer is implicit
                                "--outputmode", om,
                                "--activation", BEST_ACTIVATION,
                                '--fouriercount', str(BEST_FOURIER_COUNT),
                                '--fourierstd', str(BEST_FOURIER_STD),
                                '--volumetric_features_channels', str(BEST_GRID_CHANNELS),
                                '--volumetric_features_resolution', str(BEST_GRID_RESOLUTION),
                                "--lossmode", "densitygrad",
                                "-%s"%ld, "1",
                                "--gradient_l1", "1" if lg=='l1' else "0",
                                "--gradient_l2", "1" if lg == 'l2' else "0",
                                "--gradient_weighting", str(_gradient_weight(weight_index)),
                                '-lr', '0.01',
                                "--lr_step", "100",
                                "-i", str(epochs),
                                "--logdir", BASE_PATH + '/log',
                                "--modeldir", BASE_PATH + '/model',
                                "--hdf5dir", BASE_PATH + '/hdf5',
                                '--save_frequency', '20',
                                '--seed', str(seed),
                            ]
                            runs.append(Run(
                                config = config,
                                filename=filename,
                                do_rebuild=importance>0,
                                rebuild_importance = importance,
                                rebuild_use_color= color,
                                seed = seed,
                                network_output_mode=om,
                                loss_density=ld,
                                loss_gradient=lg,
                                gradient_weight_index=weight_index,
                                args=args
                            ))

    return runs

def train(config: Config, runs: List[Run]):
    def run(args, filename):
        args2 = args + ["--name", filename]
        if os.path.exists(os.path.join(BASE_PATH, 'hdf5', filename+".hdf5")):
            print("Skipping", filename)
        else:
            print("\n=====================================\nRun", filename)
            subprocess.run(args2, check=True)

    print("Number of runs:", len(runs))
    shuffled = list(runs)
    random.shuffle(shuffled)
    for r in shuffled:
        run(r.args, r.filename)

def create_scripts(config: Config, runs: List[Run]):
    script_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../experiments/open"))
    print("script_dir:", script_dir)

    print("Number of runs:", len(runs))
    shuffled = list(runs)
    random.seed(42)
    random.shuffle(shuffled)
    for i,r in enumerate(shuffled):
        output_filename_sh = os.path.join(script_dir, "gradients-%05d-"%i + r.filename + ".sh")
        args = " ".join(["python"] + r.args[1:] + ["--name", r.filename]) # replace path to python for the server
        with open(output_filename_sh, "w") as f:
            f.write(args)

def _eval_world(interp: pyrenderer.VolumeInterpolationNetwork,
                dataloader: torch.utils.data.DataLoader):
    device = torch.device('cuda')

    density_l1 = None
    density_l2 = None
    gradient_l1 = None
    gradient_l2 = None
    gradient_length_l1 = None
    gradient_cosine_simX = [None] * len(EVAL_LENGTH_THRESHOLDS)
    weights = None

    def append(out, v):
        v = v.cpu().numpy()
        if out is None: return v
        return np.concatenate((out, v), axis=0)

    with torch.no_grad():
        for locations_gt, densities_gt, gradients_gt, opacities_gt in tqdm.tqdm(dataloader):
            locations_gt = locations_gt[0].to(device=device)
            densities_gt = densities_gt[0].to(device=device)
            gradients_gt = gradients_gt[0].to(device=device)
            opacities_gt = opacities_gt[0].to(device=device)

            densities_pred, gradients_pred = interp.evaluate_with_gradients(locations_gt)

            density_l1 = append(density_l1, torch.abs(densities_gt - densities_pred)[:, 0])
            density_l2 = append(density_l2, F.mse_loss(densities_gt, densities_pred, reduction='none')[:, 0])
            weights = append(weights, opacities_gt[:, 0])

            gradient_l1 = append(gradient_l1,
                                 torch.mean(torch.abs(gradients_gt - gradients_pred), dim=1))
            gradient_l2 = append(gradient_l2,
                                 torch.mean(F.mse_loss(gradients_gt, gradients_pred, reduction='none'), dim=1))
            len_gt = torch.linalg.norm(gradients_gt, dim=1, keepdim=True)
            len_pred = torch.linalg.norm(gradients_pred, dim=1, keepdim=True)
            gradient_length_l1 = append(gradient_length_l1, torch.abs(len_gt - len_pred)[:, 0])
            len_gt = torch.clip(len_gt, min=1e-5)
            len_pred = torch.clip(len_pred, min=1e-5)
            N = gradients_gt.shape[0]
            cosine_sim = torch.bmm((gradients_gt / len_gt).reshape(N, 1, 3),
                                   (gradients_pred / len_pred).reshape(N, 3, 1))
            cosine_sim = cosine_sim[:, 0, 0]
            len_gt = len_gt[:, 0]

            for i in range(len(EVAL_LENGTH_THRESHOLDS)):
                length_mask = len_gt > EVAL_LENGTH_THRESHOLDS[i]
                cosine_sim_filtered = torch.masked_select(cosine_sim, length_mask)
                gradient_cosine_simX[i] = append(gradient_cosine_simX[i], cosine_sim_filtered)

    def extract_stat(v, weights):
        EVAL_WEIGHT_EPSILON = 1e-3
        d = {
            'min': float(np.min(v)),
            'max': float(np.max(v)),
            'mean': float(np.mean(v)),
            'median': float(np.median(v)),
            'std': float(np.std(v)),
        }
        if weights is not None:
            d['weighted_average'] = float(np.average(v, weights=weights))
            d['min_masked'] = float(np.min(v[weights > EVAL_WEIGHT_EPSILON]))
            d['max_masked'] = float(np.max(v[weights > EVAL_WEIGHT_EPSILON]))
        return d

    return {
        'density_l1': extract_stat(density_l1, weights),
        'density_l2': extract_stat(density_l2, weights),
        'gradient_l1': extract_stat(gradient_l1, weights),
        'gradient_l2': extract_stat(gradient_l2, weights),
        'length_l1': extract_stat(gradient_length_l1, weights),
        'cosine_similarity': [
            {'threshold': EVAL_LENGTH_THRESHOLDS[i], 'data': extract_stat(gradient_cosine_simX[i], None)}
            for i in range(len(EVAL_LENGTH_THRESHOLDS))
        ],
    }

def eval(config: Config, runs: List[Run]) -> str:
    """
    For all runs, compute the statistics, save them in a
    per-run file for caching and assemble all such stats in a big table
    that is returned.

    A lot of that code is duplicated and modified from
    eval_GradientNetworks.py
    :param config:
    :param runs:
    :return:
    """

    print("Evaluate")
    statistics_file_json = os.path.join(BASE_PATH, 'stats-%s.json' % config.name)
    #statistics_file_csv = os.path.join(BASE_PATH, 'stats-%s.csv' % config.name)
    if os.path.exists(statistics_file_json):
        print("Statistics file already exists!")
        return statistics_file_json
    global_statistics = []

    device = torch.device('cuda')
    timer = pyrenderer.GPUTimer()
    rendering_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED

    # world
    num_points = EVAL_WORLD_NUM_POINTS  # 256**3 #512**3
    batch_size = min(EVAL_WORLD_NUM_POINTS, 128 ** 3)
    num_batches = num_points // batch_size

    # screen
    width = EVAL_SCREEN_SIZE
    height = EVAL_SCREEN_SIZE
    stepsize = DEFAULT_STEPSIZE
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    # reference
    base_ln = None
    screen_color_reference = None
    screen_normal_reference = None
    screen_camera = None
    world_dataloader = None

    os.makedirs(os.path.join(BASE_PATH, 'stats'), exist_ok=True)
    image_folder = os.path.join(BASE_PATH, "images")
    os.makedirs(image_folder, exist_ok=True)
    # For each run:
    for i,run in enumerate(runs):
        base_filename = run.filename
        hdf5_filename = os.path.abspath(os.path.join(BASE_PATH, 'hdf5', base_filename + ".hdf5"))
        stat_filename = os.path.abspath(os.path.join(BASE_PATH, 'stats', base_filename + ".json"))

        if os.path.exists(stat_filename):
            print(f"Stats for run {base_filename} already computed, skip")
            # append to global stats file
            with open(stat_filename, 'r') as f:
                global_statistics.append(json.load(f))
            continue
        local_stats = {
            'filename': run.filename,
            'do_rebuild': run.do_rebuild,
            'rebuild_importance': run.rebuild_importance,
            'rebuild_use_color': run.rebuild_use_color,
            'seed': run.seed,
            'network_output_mode': run.network_output_mode,
            'loss_density': run.loss_density,
            'loss_gradient': run.loss_gradient,
            'gradient_weight_index': run.gradient_weight_index
        }

        # load network
        if not os.path.exists(hdf5_filename):
            print(f"HDF5-file for run {base_filename} not found, skip")
            continue
        try:
            ln = LoadedModel(hdf5_filename, force_config_file=config.settings)
            ln.save_compiled_network(hdf5_filename.replace('.hdf5', '.volnet'))
            volume_interp_network = ln.get_volume_interpolation_network()
            volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.OFF_OR_DIRECT
        except Exception as e:
            print("Unable to load network for run '%s':" % base_filename, e)
            continue
        print(f"Network loaded for run {base_filename} (run {i+1} / {len(runs)})")
        if base_ln is None:
            base_ln = ln # use this as baseline (only the settings are important here)

        # EVALUATE SCREEN
        # reference
        if screen_color_reference is None:
            screen_camera = base_ln.get_default_camera()
            screen_color_reference = base_ln.render_reference(screen_camera, width, height, timer=None,
                channel=pyrenderer.IImageEvaluator.Color)
            imageio.imwrite(
                os.path.join(image_folder, '%s-color-reference.png' % config.name),
                LoadedModel.convert_image(screen_color_reference))
            screen_normal_reference = base_ln.render_reference(screen_camera, width, height, timer=None,
                 channel=pyrenderer.IImageEvaluator.Normal)
            imageio.imwrite(
                os.path.join(image_folder, '%s-normal-reference.png' % config.name),
                LoadedModel.convert_image(screen_normal_reference))
        # evaluate
        with torch.no_grad():
            ln.render_network(
                screen_camera, width, height, rendering_mode, stepsize, timer=None,
                channel=pyrenderer.IImageEvaluator.Color)  # warmup
            color_image = ln.render_network(
                screen_camera, width, height, rendering_mode, stepsize, timer=timer,
                channel=pyrenderer.IImageEvaluator.Color)  # actual rendering
            color_imgname = os.path.join(image_folder, '%s-color-%s.png' % (config.name, run.filename))
            imageio.imwrite(
                color_imgname,
                LoadedModel.convert_image(color_image))
            # normal image
            normal_image = ln.render_network(
                screen_camera, width, height, rendering_mode, stepsize, timer=None,
                channel=pyrenderer.IImageEvaluator.Normal)
            normal_imgname = os.path.join(image_folder, '%s-normal-%s.png' % (config.name, run.filename))
            imageio.imwrite(
                normal_imgname,
                LoadedModel.convert_image(normal_image))
            # return stats
            local_stats['screen'] = {
                "time_seconds": timer.elapsed_milliseconds() / 1000.0,
                "color_ssim": ssim_loss(color_image, screen_color_reference).item(),
                "color_lpips": lpips_loss(color_image, screen_color_reference).item(),
                "normal_ssim": ssim_loss(normal_image, screen_normal_reference).item(),
                "normal_lpips": lpips_loss(normal_image, screen_normal_reference).item(),
                'color_image_path': color_imgname,
                'normal_image_path': normal_imgname
            }

        # EVALUATE WORLD
        if world_dataloader is None:
            print("Create world dataset")
            dataset = []
            volume_interpolation = base_ln.get_image_evaluator().volume
            ray_evaluator = base_ln.get_image_evaluator().ray_evaluator
            min_density = ray_evaluator.min_density
            max_density = ray_evaluator.max_density
            tf_evaluator = ray_evaluator.tf
            sampler = PlasticSampler(3)
            for i in tqdm.trange(num_batches):
                indices = np.arange(i * batch_size, (i + 1) * batch_size, dtype=np.int32)
                locations = sampler.sample(indices).astype(np.float32)
                locations_gpu = torch.from_numpy(locations).to(device=device)
                densities, gradients = volume_interpolation.evaluate_with_gradients(locations_gpu)
                colors = tf_evaluator.evaluate(
                       densities, min_density, max_density, gradients=gradients)
                opacities = colors[:,3:4]
                dataset.append((
                    locations,
                    torch.clamp(densities, 0.0, 1.0).cpu().numpy(),
                    gradients.cpu().numpy(),
                    opacities.cpu().numpy()))
            world_dataloader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False)

        # evaluate
        volume_interp_network = ln.get_volume_interpolation_network()
        local_stats['world'] = _eval_world(
            volume_interp_network, world_dataloader)

        # save local stats and append to the global stats file
        with open(stat_filename, 'w') as f:
            json.dump(local_stats, f)
        global_statistics.append(local_stats)

    # save statistics
    print("\n===================================== Done, save statistics")
    with open(statistics_file_json, "w") as f:
        json.dump(global_statistics, f)
    return statistics_file_json


def make_plots1(config: Config, runs: List[Run], statistics_file: str):
    sns.set_theme(style="whitegrid", palette="muted")
    with open(statistics_file, "r") as f:
        global_statistics = json.load(f)

    STATISTICS = [
        ("SSIM-Normal $\\rightarrow$", ["screen", "normal_ssim"]),
        ("LPIPS-Normal $\\leftarrow$", ["screen", "normal_lpips"]),
        ("SSIM-Color $\\rightarrow$", ["screen", "color_ssim"]),
        ("SSIM-Normal $\\leftarrow$", ["screen", "color_lpips"]),
        ("avg$(|n|_2^2)$ $\\leftarrow$", ["world", "gradient_l2", "mean"]),
        ("avg$(|n|_2^2)$ weighted $\\leftarrow$", ["world", "gradient_l2", "weighted_average"])
    ]
    PARAMETERS = [
        ("Gradient Weight", "gradient_weight_index"),
        ("Network Output", "network_output_mode"),
        ("Density-Loss", "loss_density"),
        ("Gradient-Loss", "loss_gradient"),
        ("Resampling", [("$p$", "rebuild_importance"), ("$c$", "rebuild_use_color")])
    ]

    num_cols = len(STATISTICS)
    num_rows = len(PARAMETERS)
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, sharex="col", sharey='row', figsize=(5*num_cols, 4*num_rows))
    fig.suptitle(config.name)
    # column labels
    for col in range(num_cols):
        axes[0, col].set_title(STATISTICS[col][0])
    # process parameter by parameter
    for row in range(num_rows):
        # labels
        axes[row, 0].set_ylabel(PARAMETERS[row][0])
        # figure out possible value ranges and collect data
        # we make use here that the values come in sorted due to how the runs are constructed
        keys = set()
        keys_list = list()
        values = defaultdict(lambda: [[] for _ in range(num_cols)])
        parameter = PARAMETERS[row][1]
        for local_stat in global_statistics:
            # extract key
            if isinstance(parameter, str):
                key = local_stat[parameter]
            else: # list of sub-keys
                key = []
                for n, v in parameter:
                    key.append(f"{n}={local_stat[v]}")
                key = ", ".join(key)
            if not key in keys:
                keys.add(key)
                keys_list.append(key)
            # fill values
            local_values = values[key]
            for col in range(num_cols):
                param_path = STATISTICS[col][1]
                value = local_stat
                for p in param_path:
                    value = value[p]
                local_values[col].append(value)
        # make plots
        for col in range(num_cols):
            ax = axes[row, col]
            data = [values[key][col] for key in keys_list]
            sns.stripplot(data=data, orient='h', size=2, ax=ax)
            ax.set_yticklabels(keys_list)

    fig.tight_layout()
    output_filename = os.path.join(BASE_PATH, f'GradientNetworks2_{config.name}_Swarmplot')
    fig.savefig(output_filename+'.pdf', bbox_inches='tight')
    fig.savefig(output_filename+'.png', bbox_inches='tight')

def make_plots2(config: Config, runs: List[Run], statistics_file: str):
    sns.set_theme(style="whitegrid", palette="muted")
    with open(statistics_file, "r") as f:
        global_statistics = json.load(f)

    STATISTICS = [
        ("LPIPS-Color", "LPIPS-Color $\\downarrow$", ["screen", "color_lpips"]),
        ("LPIPS-Normal", "LPIPS-Normal $\\downarrow$", ["screen", "normal_lpips"]),
        ("GradientL2", "avg$(|n|_2^2)$ $\\downarrow$", ["world", "gradient_l2", "mean"]),
    ]

    PARAMETER_X_COARSE = ("Network Output", "network_output_mode")
    PARAMETER_X_FINE = ("Gradient Weight", [("$w$", "gradient_weight_index")])
    PARAMETER_Y_COARSE = ("Resampling", [("$p$", "rebuild_importance"), ("$c$", "rebuild_use_color")])
    PARAMETER_Y_FINE = ("Gradient-Loss", "loss_gradient")

    parameterXCoarseKeys = []
    parameterXFineKeys = []
    parameterYCoarseKeys = []
    parameterYFineKeys = []

    # Pass 1: extract value ranges
    for local_stat in global_statistics:
        for (name, parameter), keys, index_key in [
            (PARAMETER_X_COARSE, parameterXCoarseKeys, "index-xcoarse"),
            (PARAMETER_X_FINE, parameterXFineKeys, "index-xfine"),
            (PARAMETER_Y_COARSE, parameterYCoarseKeys, "index-ycoarse"),
            (PARAMETER_Y_FINE, parameterYFineKeys, "index-yfine")
        ]:
            if isinstance(parameter, str):
                key = local_stat[parameter]
            else: # list of sub-keys
                key = []
                for n, v in parameter:
                    key.append(f"{n}={local_stat[v]}")
                key = ", ".join(key)
            if not key in keys:
                keys.append(key)
            index = keys.index(key)
            local_stat[index_key] = index

    # Pass 2: for each statistic, extract values
    for stat_index, (stat_filename, stat_humanname, param_path) in enumerate(STATISTICS):
        heatmap_values = [[np.zeros((len(parameterYFineKeys), len(parameterXFineKeys)), dtype=np.float32)
                           for y in range(len(parameterYCoarseKeys))] for x in range(len(parameterXCoarseKeys))]
        heatmap_counts = [[np.zeros((len(parameterYFineKeys), len(parameterXFineKeys)), dtype=np.float32)
                           for y in range(len(parameterYCoarseKeys))] for x in range(len(parameterXCoarseKeys))]
        heatmap_individual_values = defaultdict(list)
        # fill values
        min_value = 1e10
        max_value = 0
        for local_stat in global_statistics:
            value = local_stat
            for p in param_path:
                value = value[p]
            max_value = max(max_value, value)
            min_value = min(min_value, value)
            heatmap_values[local_stat['index-xcoarse']][local_stat['index-ycoarse']][local_stat['index-yfine'],local_stat['index-xfine']] += value
            heatmap_counts[local_stat['index-xcoarse']][local_stat['index-ycoarse']][local_stat['index-yfine'],local_stat['index-xfine']] += 1
            heatmap_individual_values[(local_stat['index-xcoarse'], local_stat['index-ycoarse'], local_stat['index-xfine'], local_stat['index-yfine'])].append((value, local_stat))
        max_entries_per_cell = max([len(v) for v in heatmap_individual_values.values()])

        # make figure
        fig, axes = plt.subplots(nrows=len(parameterYCoarseKeys), ncols=len(parameterXCoarseKeys),
                                 sharex='col', sharey='row', figsize=(14, 12))
        fig.suptitle(f"{config.name}: {stat_humanname}")
        cbar_ax = fig.add_axes([1.01, .2, .03, .6])

        norm = matplotlib.colors.LogNorm(min_value, max_value)
        cmap = "rocket_r"

        dot_size = 100 / max_entries_per_cell
        dot_offset = 0.3
        for x in range(len(parameterXCoarseKeys)):
            for y in range(len(parameterYCoarseKeys)):
                values = heatmap_values[x][y]
                counts = heatmap_counts[x][y]
                ax = axes[y,x]
                # normalize (produces NaNs for empty cells)
                values /= counts
                # heatmap
                sns.heatmap(values, ax=ax,
                            norm=norm, cmap=cmap,
                            annot=True, fmt='.3f',
                            linewidths=1, square=True,
                            xticklabels=parameterXFineKeys,
                            yticklabels=parameterYFineKeys,
                            cbar=(x==0 and y==0),
                            cbar_ax=cbar_ax if (x==0 and y==0) else None)
                # annotate with individual keys
                for xfine,yfine in itertools.product(range(len(parameterXFineKeys)), range(len(parameterYFineKeys))):
                    vx = heatmap_individual_values[(x, y, xfine, yfine)]
                    offX = np.linspace(xfine + 0.5 - dot_offset, xfine + 0.5 + dot_offset, len(vx), endpoint=True)
                    offY = np.array([yfine + 0.8]*len(vx))
                    ax.scatter(offX, offY, s=dot_size, c=[v for v,stat in vx], cmap=cmap, norm=norm, edgecolors='k')
                # highlight certain values
                for h in EVAL_HIGHLIGHTS:
                    if not h.output_mode == parameterXCoarseKeys[x]: continue
                    if not h.dataset_importance_key == parameterYCoarseKeys[y]: continue
                    xfine = parameterXFineKeys.index("$w$=%d"%h.gradient_weight)
                    yfine = parameterYFineKeys.index(h.loss_gradient)
                    offX = 0.5 + xfine
                    offY = 0.5 + yfine
                    rect = patches.Rectangle((offX-0.5, offY-0.5), 1.0, 1.0, linewidth=5, edgecolor=h.color, fill=False)
                    rect.set_clip_on(False)
                    ax.add_patch(rect)
                    # save image of best run for that stat
                    if stat_index == 0:
                        vx = heatmap_individual_values[(x, y, xfine, yfine)]
                        if len(vx)==0:
                            print("WARNING: no run for the highlighted configuration found")
                        else:
                            best_index = min(range(len(vx)), key=lambda i: vx[i][0])
                            for channel in ['normal', 'color']:
                                filename = f"{config.name}-{channel}-{vx[best_index][1]['filename']}.png"
                                output_filename = f'GradientNetworks2_{config.name}_{channel}_{h.color}.png'
                                shutil.copy(os.path.join(BASE_PATH, "images", filename), os.path.join(BASE_PATH, output_filename))

                                filename = f"{config.name}-{channel}-reference.png"
                                output_filename = f'GradientNetworks2_{config.name}_{channel}_reference.png'
                                shutil.copy(os.path.join(BASE_PATH, "images", filename),os.path.join(BASE_PATH, output_filename))
                # labels
                if x==0:
                    # hack: special label for disabled resampling
                    if parameterYCoarseKeys[y] == "$p$=0, $c$=False":
                        ax.set_ylabel("No resampling")
                    else:
                        ax.set_ylabel(parameterYCoarseKeys[y])
                if y==len(parameterYCoarseKeys)-1:
                    ax.set_xlabel(parameterXCoarseKeys[x])

        fig.tight_layout()
        output_filename = os.path.join(BASE_PATH, f'GradientNetworks2_{config.name}_{stat_filename}')
        fig.savefig(output_filename+".pdf", bbox_inches='tight')
        fig.savefig(output_filename+".png", bbox_inches='tight')

if __name__ == '__main__':
    main()