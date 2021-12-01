"""
The plots for the paper.
Requires eval_TimeVolumetricFeatures.py and
eval_TimeVolumetricFeatures2.py to be executed beforehand.
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import re
import numpy as np
import shutil
import subprocess
import itertools
import imageio
import json
import torch
import io
import tqdm
from collections import OrderedDict
from typing import Callable, NamedTuple, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

BASE_PATH = 'volnet/results/eval_TimeVolumetricFeatures'
OUTPUT = 'TimeVolumetricFeatures-PlotForPaper.eps'

class Config(NamedTuple):
    stats_file: str
    cfg_filter: str
    cfg: str
    key_frames: Tuple[int,int,int]
    train_frames: Tuple[int,int,int]

configX = [
    Config('stats.json', '.*plumeEnsemble.*', 'plume-time', (30,101,10), (30,101,5)),
    Config('stats-time2.json', '.*plumeEnsemble.*', 'plume-time2', (30,101,10), (30,101,2))
]

class Variant(NamedTuple):
    key: str
    human_name: str
    color: str

variantX = [
    Variant('-l32x4-fNeRF14-G32C16-none', 'Grid: $32^3*16$, Time: none', 'darkblue'),
    Variant('-l32x4-fNeRF14-G32C16-direct', 'Grid: $32^3*16$, Time: direct', 'mediumslateblue'),
    Variant('-l32x4-fNeRF14-G32C16-fourier', 'Grid: $32^3*16$, Time: fourier', 'darkorange'),
    Variant('-l32x4-fNeRF14-G32C16-both', 'Grid: $32^3*16$, Time: both', 'peru'),
    Variant('-l32x4-fNeRF14-G32C16-steady-direct', 'Grid: $32^3*16$, fixed in time', 'red'),
    Variant('-l0c0-G32C16', 'Grid: $32^3*16$, no network', 'darkgreen'),
    Variant('-l0c0-G178C1', 'Grid: $178^3*1$, no network', 'mediumseagreen')
]

def make_plots():
    plt_height = 300
    plt_width = 1000
    plt_dpi = 96

    # load stats
    stats = []
    for cfg in configX:
        with open(os.path.join(BASE_PATH, cfg.stats_file), 'r') as f:
            stats.append(json.load(f))

    # prepare scales and colors
    def invLogForward(x):
        return 1-np.log(1-np.clip(x, a_min=0, a_max=0.999))
    def invLogInverse(y):
        return 1-np.exp(1-y)

    def set_scale_ssim(ax):
        ax.set_yscale('function', functions=(invLogForward, invLogInverse))
        ax.set_yticks([1.0, 0.99, 0.9, 0.8, 0.5], minor=False)
        ax.set_yticks(
            [0.99 + 0.001*i for i in range(10)] +
            [0.9 + 0.01*i for i in range(10)] +
            [0.5 + 0.1*i for i in range(5)],
            minor=True)

    statName = 'SSIM $\\rightarrow$'
    statTag = "ssim"
    ensemble = 0
    fig, axs = plt.subplots(nrows=len(configX), ncols=1, sharex=True,
                            figsize=(plt_width / plt_dpi, len(configX) * plt_height / plt_dpi),
                            dpi=plt_dpi)
    handles = []
    handle_names = []
    for row, (cfg, stat) in enumerate(zip(configX, stats)):
        kmin, kmax, kstep = cfg.key_frames
        X = list(range(kmin, kmax))
        Xkeyframes = list(range(*cfg.key_frames))
        Xtrain = list(range(*cfg.train_frames))

        # find local stats with that config
        localStats = None
        pattern = re.compile(cfg.cfg_filter)
        for s in stat:
            if pattern.fullmatch(s['cfg'][0]):
                localStats = s
                break
        assert localStats is not None

        if row == len(configX) - 1:
            axs[row].set_xlabel("Time")
        axs[row].set_ylabel(statName)
        for x in Xtrain:
            axs[row].axvline(x, ls='--', lw=1, color='gray')
        for x in Xkeyframes:
            axs[row].axvline(x, ls='-', lw=1.2, color='gray')

        for i,variant in enumerate(variantX):
            key = "TimeVolumetricLatentSpace2-" + cfg.cfg + variant.key
            num_cameras = localStats[key]['num_cameras']
            Y = np.array([np.mean(
                [localStats[key][statTag][ensemble][t][c] for c in range(num_cameras)]) for t in
                 range(len(X))])
            h = axs[row].plot(X, Y, 'o-', linewidth=1, markersize=2,
                              color=variant.color)
            if row == 0:
                handles.append(h[0])
                handle_names.append(variant.human_name)
                c = matplotlib.colors.to_rgba(variant.color)
                c_latex = f'\\definecolor{{color{chr(ord("a")+i)}}}{{rgb}}{{{c[0]}, {c[1]}, {c[2]}}} % {variant.human_name}'
                print(f"Color of {variant.human_name}: {c_latex}")

        set_scale_ssim(axs[row])
        t = axs[row].annotate(chr(ord('a')+row)+')',
                              xy=(5,5), xycoords='axes points',
                              fontsize='large')

    lgd = fig.legend(
        handles, handle_names,
        # bbox_to_anchor=(0.75, 0.7), loc='lower center', borderaxespad=0.
        loc='center left', bbox_to_anchor=(0.9, 0.5),
        ncol=1)
    fig.subplots_adjust(hspace=+0.05)
    figure_path = os.path.join(BASE_PATH, OUTPUT)
    fig.savefig(figure_path, bbox_inches='tight', bbox_extra_artists=(lgd,))
    print("Figure saved to", figure_path)


if __name__ == '__main__':
    make_plots()