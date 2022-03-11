
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
import tqdm
import random
from collections import OrderedDict
from typing import Callable, NamedTuple, Tuple, List, Optional, Any, Union
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.cm
import matplotlib.colors
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict


BASE_PATH = 'volnet/results/eval_CompressionExtended'
IMAGE_FOLDER_LATEX = os.path.join(BASE_PATH, "images_latex")

class Config(NamedTuple):
    name: str
    settings_train: str
    settings_eval: str
    human_name: str
    dtype: str
    samples: str
    epochs: int
    target_compression_for_vis: int # images for the case closest to that compression rate are exported
    num_refinement: int = 0
    use_chunks: bool = False
    overwrite_checkpoints: Optional[int] = None
    overwrite_rebuild: Optional[int] = None
    requires_gradient: Optional[bool] = False
    fvsrn_extra_resolutions: Optional[List[int]] = None
    fvsrn_force_resolutions: Optional[List[str]] = None # for non-cube resolutions
    neurcomp_extra_channels: Optional[List[int]] = None
    tthresh_extra_values: Optional[List[int]] = None
    cudacompress_extra_quant: Optional[List[float]] = None
    force_chunks_per_axis: Optional[Union[Tuple[int,int,int], int]] = None # forces the number of chunks per axis (cudaCompress+TThres)

configX = [
    Config(
        name = "jet",
        settings_eval = "config-files/LuBerger-Jet-v3-shaded.json",
        settings_train = "config-files/LuBerger-Jet-v3-shaded.json",
        human_name="Jet",
        dtype='float',
        samples= "512**3",
        target_compression_for_vis=1000,
        num_refinement=1, #31, #-> 32*8 = 256 samples per pixel
        epochs=100, # otherwise, it takes too long
        requires_gradient=True,
        tthresh_extra_values=[20],
    ),
    Config(
        name="ejecta1024",
        settings_train="config-files/ejecta1024-v6-dvr.json",
        settings_eval="config-files/ejecta1024-v6-dvr.json",
        human_name="Ejecta $1024^3$",
        dtype='ushort', # unsigned short
        samples="1024**3",
        target_compression_for_vis=10000,
        epochs=40,  # otherwise, it takes too long
        overwrite_checkpoints=5,
        overwrite_rebuild=21,
        use_chunks=True
    ),
    Config(
        name="rm1024",
        settings_train="config-files/RM1024-v2-dvr.json",
        settings_eval="config-files/RM1024-v2-dvr.json",
        human_name="RM $1024^3$",
        dtype='ubyte',
        samples="1024**3",
        target_compression_for_vis=1000,
        epochs=40,  # otherwise, it takes too long
        overwrite_checkpoints=5,
        overwrite_rebuild=21,
        use_chunks=True,
        fvsrn_extra_resolutions=[96, 128, 160],
        neurcomp_extra_channels=[386],
        tthresh_extra_values=[15, 20],
        cudacompress_extra_quant=[0.2, 0.5],
    ),
    Config(
        name="miranda",
        settings_train="config-files/Miranda-v1-dvr.json",
        settings_eval="config-files/Miranda-v1-dvr.json",
        human_name="Miranda $512^3$",
        dtype='float', # unsigned short
        samples="512**3",
        target_compression_for_vis=10000,
        epochs=100,  # otherwise, it takes too long
        use_chunks=True,
        tthresh_extra_values=[15, 20],
        cudacompress_extra_quant=[0.2, 0.5],
    ),
    ]

# SETTINGS cudaCompress
CUDA_COMPRESS_LEVEL = 2
CUDA_COMPRESS_QUANT_STEP_X = [0.001, 0.005, 0.01, 0.05, 0.1]
CUDA_COMPRESS_BEST_CHUNKSIZE = 256

# SETTINGS TThresh
TTHRESH_STAT = 'PSNR'
TTHRESH_VALUE_X = [30, 40, 45, 50, 60]
TTHRESH_BEST_CHUNKSIZE = 256

# SETTINGS fV-SRN
FVSRN_NETWORK_LAYERS = 4
FVSRN_NETWORK_CHANNELS = 32
FVSRN_ACTIVATION = "SnakeAlt:1"
FVSRN_FOURIER_STD = -1 # NERF
FVSRN_FOURIER_COUNT = 14
FVSRN_GRID_CHANNELS = 16
FVSRN_GRID_RESOLUTION_X = [16, 24, 32, 48, 64]

# SETTINGS neurcomp / LuBerger
NEURCOMP_ACTIVATION = "ResidualSine"
NEURCOMP_RESIDUAL_BLOCKS = 8
NEURCOMP_CHANNELS_X = [32, 64, 96, 128, 160, 192, 256]
NEURCOMP_AUTOGRAD_ARGS = {"batchsize": 256**2}

DEFAULT_CHECKPOINTS = 50
DEFAULT_REBUILD = 51

width = 1024
height = 1024
num_points = 10 ** 6

class Case(NamedTuple):
    algorithm: str # fV-SRN or neurcomp
    filename: str # path to the HDF5-file containing the training result
    args: List[str]
    max_batchsize: int = None # for large neurcomp, to not run out of memory

def _collect_cases(config: Config):
    save_frequency = config.overwrite_checkpoints or 20
    rebuild = config.overwrite_rebuild or 51
    common_args = [
        sys.executable, "volnet/train_volnet.py",
        config.settings_train,
        "--train:mode", "world",
        "--train:samples", config.samples,
        "--train:sampler_importance", "0.01",
        '--rebuild_dataset', str(rebuild),
        "--val:copy_and_split",
        "--outputmode", "density:direct",
        "--lossmode", "density",
        "-l1", "1",
        "--lr_step", "100",
        "-i", str(config.epochs),
        "--logdir", BASE_PATH + '/log',
        "--modeldir", BASE_PATH + '/model',
        "--hdf5dir", BASE_PATH + '/hdf5',
        '--save_frequency', str(save_frequency)
    ]
    cases = []
    # fV-SRN
    if config.fvsrn_force_resolutions is not None:
        assert config.fvsrn_extra_resolutions is None, "Already forcing the fV-SRN resolutions, extra resolutions are ignored"
        rx = config.fvsrn_force_resolutions
    else:
        rx = FVSRN_GRID_RESOLUTION_X + (config.fvsrn_extra_resolutions or [])
    for resolution in rx:
        filename = config.name + "-fvSRN-%s" % str(resolution)
        fvsrn_args = [
            "--layers", ':'.join([str(FVSRN_NETWORK_CHANNELS)] * (FVSRN_NETWORK_LAYERS - 1)),
            "--train:batchsize", "64*64*128",
            "--activation", FVSRN_ACTIVATION,
            '--fouriercount', str(FVSRN_FOURIER_COUNT),
            '--fourierstd', str(FVSRN_FOURIER_STD),
            '--volumetric_features_channels', str(FVSRN_GRID_CHANNELS),
            '--volumetric_features_resolution', str(resolution),
            '-lr', '0.002',
            "--name", filename
        ]
        cases.append(Case("fv-SRN", filename, common_args+fvsrn_args))
    # neurcomp
    cx = NEURCOMP_CHANNELS_X + (config.neurcomp_extra_channels or [])
    for channels in cx:
        filename = config.name + "-neurcomp_args-%d" % channels
        neurcomp_args = [
            "--layers", ':'.join([str(channels)] * (NEURCOMP_RESIDUAL_BLOCKS - 1)),
            "--train:batchsize", "64*64*32",
            "--activation", NEURCOMP_ACTIVATION,
            '--fouriercount', '0',
            '-lr', '0.00005',
            "--name", filename
        ]
        if channels > 256:
            max_batchsize = num_points*256//channels
        else:
            max_batchsize = None
        cases.append(Case("neurcomp", filename, common_args + neurcomp_args, max_batchsize))

    return cases

def write_scripts_for_training():
    script_dir = os.path.abspath(os.path.join(os.path.split(__file__)[0], "../experiments/open"))
    print("script_dir:", script_dir)
    for config in configX:
        cases = _collect_cases(config)
        print("Number of runs:", len(cases))
        shuffled = list(cases)
        random.seed(42)
        random.shuffle(shuffled)
        for i, r in enumerate(shuffled):
            output_filename_sh = os.path.join(script_dir, "Compression-%05d-" % i + r.filename + ".sh")
            args = " ".join(["python"] + r.args[1:] + ["--name", r.filename])  # replace path to python for the server
            with open(output_filename_sh, "w") as f:
                f.write(args)

def eval_and_plot():
    files = []
    for config in configX:
        cases = _collect_cases(config)
        statistics_file = _eval(config, cases)
        files.append(statistics_file)
    _plot(configX[:2], files[:2], "-Part1")
    _plot(configX[2:], files[2:], "-Part2")

def _eval(config: Config, cases: List[Case]) -> str:
    print("Evaluate")
    statistics_file = os.path.join(BASE_PATH, 'stats-%s.json' % config.name)
    if os.path.exists(statistics_file):
        print("Statistics file already exists!")
        return statistics_file

    import common.utils as utils
    import pyrenderer
    from volnet.inference import LoadedModel, ChunkedNetwork
    from losses.lossbuilder import LossBuilder
    from volnet.raytracing import Raytracing
    from volnet.bricking import create_chunks, reassemble_chunks, BrickedRendering

    num_cameras = 1  # we only show one image
    timer = pyrenderer.GPUTimer()
    enable_preintegration = True
    VERBOSE_BASELINES = True

    output_stats = {
        "name": config.name,
        "settings": config.settings_eval,
    }
    case_stats = []
    device = torch.device('cuda')

    def psnr(p0, p1, peak=1.):
        return 10 * np.log10(peak ** 2 / np.mean((1. * p0 - 1. * p1) ** 2))
    ssim_loss = LossBuilder(device).ssim_loss(4)
    lpips_loss = LossBuilder(device).lpips_loss(4, 0.0, 1.0)

    image_folder = os.path.join(BASE_PATH, "images_%s" % config.name)
    os.makedirs(image_folder, exist_ok=True)
    base_ln = None
    camera = None
    reference_image = None
    reference_densities = None
    stepsize_world = None
    reference_volume_float = None
    reference_volume_double = None
    sample_locations = torch.rand(num_points, 3, dtype=torch.float32, device=device)

    # === fV-SRN + neurcomp ===
    def load_and_save(case:Case) -> Tuple[LoadedModel, int, LoadedModel.EvaluationMode]:
        filename = os.path.join(BASE_PATH, "hdf5", case.filename+".hdf5")
        filename = os.path.abspath(filename)
        if not os.path.exists(filename):
            print("File not found:", filename, file=sys.stderr)
            return None, 0, None
        try:

            ln = LoadedModel(
                filename,
                grid_encoding=pyrenderer.SceneNetwork.LatentGrid.ByteLinear,
                force_config_file=config.settings_eval,
                max_batchsize=case.max_batchsize)
            if enable_preintegration:
                ln.enable_preintegration(True, convert_to_texture=True)
            volnet_file = filename.replace('.hdf5', '.volnet')
            if ln.save_compiled_network(volnet_file):
                # compressed size is the size of the volnet file
                compressed_size = os.path.getsize(volnet_file)
                render_mode = LoadedModel.EvaluationMode.TENSORCORES_MIXED
            else:
                # pickle the 16bit network
                net = ln.get_network_pytorch16()
                net = ChunkedNetwork.unwrap(net)
                buffer = io.BytesIO()
                torch.save(net, buffer)
                buffer.seek(0, 2)
                compressed_size = buffer.tell()
                render_mode = LoadedModel.EvaluationMode.PYTORCH16
            return ln, compressed_size, render_mode
        except Exception as e:
            print("Unable to load '%s':" % filename, e)
            return None, 0, None

    def eval_network(case: Case, ln, compressed_size, render_mode):
        stat_filename = os.path.join(image_folder, '%s.json' % case.filename)
        if os.path.exists(stat_filename):
            print(f"Stats for run {case.filename} already computed, skip")
            with open(stat_filename, 'r') as f:
                return json.load(f)
        # evaluate world, with warmup
        print("Evaluate ", case.filename)
        _ = ln.evaluate(sample_locations, render_mode)
        densities = ln.evaluate(sample_locations, render_mode, timer=timer)
        time_world = timer.elapsed_milliseconds()
        psnr_densities = float(psnr(reference_densities.cpu().numpy(), densities.cpu().numpy()))

        # evaluate screen
        if config.requires_gradient:
            try:
                # fV-SRN
                volume_interp_network = ln.get_volume_interpolation_network()
                volume_interp_network.gradient_mode = pyrenderer.VolumeInterpolationNetwork.GradientMode.ADJOINT_METHOD
            except AttributeError:
                # Plain network
                ln.get_raytracing().set_gradient_mode(Raytracing.GradientMode.AUTODIFF)
                ln.get_raytracing().set_gradient_args(NEURCOMP_AUTOGRAD_ARGS)
        torch.cuda.reset_peak_memory_stats(device)
        mem_start = torch.cuda.max_memory_allocated(device)
        image = ln.render_network(camera, width, height, render_mode, stepsize_world, timer=timer, num_refine=config.num_refinement)
        mem_end = torch.cuda.max_memory_allocated(device)
        time_screen = timer.elapsed_milliseconds()
        ssim = float(ssim_loss(reference_image, image).item())
        lpips = float(lpips_loss(reference_image, image).item())
        image_filename = '%s.png' % case.filename
        imageio.imwrite(
            os.path.join(image_folder, image_filename),
            LoadedModel.convert_image(image))

        # write stats
        local_stats = {
            'algorithm': case.algorithm,
            'image_filename': image_filename,
            'compressed_size_bytes': compressed_size,
            'peak_memory': (mem_end-mem_start),
            'decompression_memory_bytes': compressed_size + (mem_end-mem_start),  # networks work directly on the compressed representation
            'time_world_ms': time_world,
            'time_screen_ms': time_screen,
            'psnr_world': psnr_densities,
            'ssim_screen': ssim,
            'lpips_screen': lpips
        }
        with open(stat_filename, 'w') as f:
            json.dump(local_stats, f)
        return local_stats

    for case in cases:
        ln, compressed_size, render_mode = load_and_save(case)
        if ln is None: continue

        # create reference
        if base_ln is None:
            base_ln = ln
            camera = ln.get_default_camera()
            stepsize_world = ln.get_default_stepsize()
            reference_densities = ln.evaluate_reference(sample_locations)
            reference_image = ln.render_reference(camera, width, height, stepsize_world=stepsize_world, num_refine=config.num_refinement)
            imageio.imwrite(
                os.path.join(image_folder, 'reference.png'),
                LoadedModel.convert_image(reference_image))
            # compute original volume size
            original_volume = ln.get_image_evaluator().volume.volume()
            feature = original_volume.get_feature(0)
            reference_volume_float = feature.get_level(0).to_tensor().cpu()[0]
            reference_volume_double = reference_volume_float.to(torch.float64)
            channels = feature.channels()
            resolution = feature.base_resolution()
            bytes_per_voxel = pyrenderer.Volume.bytes_per_type(feature.type())
            original_size = bytes_per_voxel * channels * \
                resolution.x * resolution.y * resolution.z
            output_stats['original_size_bytes'] = original_size
            # delete GPU memory to save space
            feature.clear_gpu_resources()
            torch.cuda.empty_cache()

            ##test
            #raw_path = "../compression/test/data/%s_%d_%d_%d.raw"%(
            #    config.name, reference_volume_float.shape[0],
            #    reference_volume_float.shape[1], reference_volume_float.shape[2])
            #with open(raw_path, "w") as f:
            #    reference_volume_float.numpy().tofile(f, sep='')
            #print("Raw file written")

        case_stats.append(eval_network(case, ln, compressed_size, render_mode))

    if base_ln is None:
        print("ERROR!! No network could be loaded")
        exit(-1)

    torch.cuda.empty_cache()
    # === cudaCompress + TThresh ===
    def eval_baseline(reconstructed_volume, filename, decompression_time, stats):
        reconstructed_volume = reconstructed_volume.to(dtype=torch.float32)
        # set volume data
        image_evaluator = base_ln.get_image_evaluator()
        volume = image_evaluator.volume.volume()
        volume_feature = volume.get_feature(0)
        volume_feature.get_level(0).from_tensor(reconstructed_volume.unsqueeze(0))
        volume_feature.get_level(0).clear_gpu_resources()
        volume_feature.get_level(0).copy_cpu_to_gpu()
        # evaluate world, with warmup
        _ = base_ln.evaluate_reference(sample_locations)
        densities = base_ln.evaluate_reference(sample_locations, timer=timer)
        time_world = timer.elapsed_milliseconds()
        psnr_densities = float(psnr(reference_densities.cpu().numpy(), densities.cpu().numpy()))

        # evaluate screen
        image = base_ln.render_reference(camera, width, height, stepsize_world=stepsize_world, timer=timer, num_refine=config.num_refinement)
        time_screen = timer.elapsed_milliseconds()
        ssim = float(ssim_loss(reference_image, image).item())
        lpips = float(lpips_loss(reference_image, image).item())
        image_filename = '%s.png' % filename
        imageio.imwrite(
            os.path.join(image_folder, image_filename),
            LoadedModel.convert_image(image))

        stats.update({
            'image_filename': image_filename,
            'time_world_ms': time_world + decompression_time,
            'time_screen_ms': time_screen + decompression_time,
            'psnr_world': psnr_densities,
            'ssim_screen': ssim,
            'lpips_screen': lpips
        })
        return stats

    # TThresh
    def _evalTThresh(v, chunked):
        filename = 'TThreshChunked-%03d'%v if chunked else 'TThresh-%03d'%v
        stat_filename = os.path.join(image_folder, '%s.json' % filename)
        if os.path.exists(stat_filename):
            print(f"Stats for run {filename} already computed, skip")
            with open(stat_filename, 'r') as f:
                d = json.load(f)
                # fix for missing image_filename-key
                if not 'image_filename' in d:
                    d['image_filename'] = filename+".png"
                return d

        print("Evaluate ", filename)
        stats = {'algorithm': 'TThreshBricked' if chunked else 'TThresh'}

        if not chunked:
            stats['num_chunks'] = 1
            compressed, stats_compression = pyrenderer.compression.compress_tthresh(
                reference_volume_double, TTHRESH_STAT, v, VERBOSE_BASELINES)
            compressed_size = len(compressed)
            stats['compressed_size_bytes'] = compressed_size
            stats['stats_compression'] = stats_compression

            reconstructed_volume, stats_decompression = pyrenderer.compression.decompress_tthresh(
                compressed, VERBOSE_BASELINES)
            stats['stats_decompression'] = stats_decompression
            stats['decompression_memory_bytes'] = \
                stats_decompression['peak_memory_cpu'] + stats_decompression['peak_memory_gpu']
            decompression_time = stats_decompression['time_ms']
        else:
            chunks_per_axis = reference_volume_double.shape[0] // TTHRESH_BEST_CHUNKSIZE
            if config.force_chunks_per_axis is not None:
                chunks_per_axis = config.force_chunks_per_axis
            chunks_in = create_chunks(reference_volume_double, chunks_per_axis)
            stats['num_chunks'] = len(chunks_in)
            compressed, stats_compression = pyrenderer.compression.compress_tthresh_chunked(
                chunks_in, TTHRESH_STAT, v, False)
            compressed_size = len(compressed)
            stats['compressed_size_bytes'] = compressed_size
            stats['stats_compression'] = stats_compression

            chunks_out, stats_decompression = pyrenderer.compression.decompress_tthresh_chunked(
                compressed, False)
            stats['stats_decompression'] = stats_decompression
            stats['decompression_memory_bytes'] = \
                stats_decompression['peak_memory_cpu'] + stats_decompression['peak_memory_gpu']
            decompression_time = stats_decompression['time_ms']
            reconstructed_volume = reassemble_chunks(reference_volume_double, chunks_per_axis, chunks_out)

        local_stats = eval_baseline(reconstructed_volume, filename, decompression_time, stats)
        with open(stat_filename, 'w') as f:
            json.dump(local_stats, f)
        return local_stats

    tthreshX = TTHRESH_VALUE_X + (config.tthresh_extra_values or [])
    for value in tthreshX:
        case_stats.append(_evalTThresh(value, False))
    for value in tthreshX:
        case_stats.append(_evalTThresh(value, True))

    # cudaCompress
    def _evalCudaCompress(qs):
        filename = 'CudaCompress-%04d' % (qs * 1000)
        stat_filename = os.path.join(image_folder, '%s.json' % filename)
        if os.path.exists(stat_filename):
            print(f"Stats for run {filename} already computed, skip")
            with open(stat_filename, 'r') as f:
                d = json.load(f)
                # fix for missing image_filename-key
                if not 'image_filename' in d:
                    d['image_filename'] = filename + ".png"
                return d

        print("Evaluate ", filename)
        stats = {'algorithm': 'cudaCompress'}

        if not config.use_chunks:
            stats['num_chunks'] = 1
            compressed, stats_compression = pyrenderer.compression.compress_cuda(
                reference_volume_float, CUDA_COMPRESS_LEVEL, qs, 1, VERBOSE_BASELINES)
            compressed_size = len(compressed)
            stats['compressed_size_bytes'] = compressed_size
            stats['stats_compression'] = stats_compression

            reconstructed_volume, stats_decompression = pyrenderer.compression.decompress_cuda(
                compressed, VERBOSE_BASELINES)
            stats['stats_decompression'] = stats_decompression
            stats['decompression_memory_bytes'] = \
                stats_decompression['peak_memory_cpu'] + stats_decompression['peak_memory_gpu']
            decompression_time = stats_decompression['time_ms']
        else:
            chunks_per_axis = reference_volume_float.shape[0] // CUDA_COMPRESS_BEST_CHUNKSIZE
            if config.force_chunks_per_axis is not None:
                chunks_per_axis = config.force_chunks_per_axis
            chunks_in = create_chunks(reference_volume_float, chunks_per_axis)
            stats['num_chunks'] = len(chunks_in)
            compressed, stats_compression = pyrenderer.compression.compress_cuda_chunked(
                chunks_in, CUDA_COMPRESS_LEVEL, qs, False)
            compressed_size = len(compressed)
            stats['compressed_size_bytes'] = compressed_size
            stats['stats_compression'] = stats_compression

            chunks_out, stats_decompression = pyrenderer.compression.decompress_cuda_chunked(
                compressed, False)
            stats['stats_decompression'] = stats_decompression
            stats['decompression_memory_bytes'] = \
                stats_decompression['peak_memory_cpu'] + stats_decompression['peak_memory_gpu']
            decompression_time = stats_decompression['time_ms']
            reconstructed_volume = reassemble_chunks(reference_volume_float, chunks_per_axis, chunks_out)

        local_stats = eval_baseline(
            reconstructed_volume, filename, decompression_time, stats)
        with open(stat_filename, 'w') as f:
            json.dump(local_stats, f)
        return local_stats

    cudaQuantX = CUDA_COMPRESS_QUANT_STEP_X + (config.cudacompress_extra_quant or [])
    for quantStep in cudaQuantX:
        case_stats.append(_evalCudaCompress(quantStep))

    # CUDA Compress with bricking during rendering
    def _evalCudaCompressBricked(qs):
        filename = 'CudaCompressBricked-%04d' % (qs * 1000)
        stat_filename = os.path.join(image_folder, '%s.json' % filename)
        if os.path.exists(stat_filename):
            print(f"Stats for run {filename} already computed, skip")
            with open(stat_filename, 'r') as f:
                d = json.load(f)
                # fix for missing image_filename-key
                if not 'image_filename' in d:
                    d['image_filename'] = filename + ".png"
                return d

        print("Evaluate ", filename)
        stats = {'algorithm': 'cudaCompressBricked'}

        chunks_per_axis = reference_volume_float.shape[0] // CUDA_COMPRESS_BEST_CHUNKSIZE
        if config.force_chunks_per_axis is not None:
            chunks_per_axis = config.force_chunks_per_axis
        br = BrickedRendering(base_ln.get_image_evaluator(), reference_volume_float)
        compressed, stats_compression, num_chunks = br.compress(chunks_per_axis, CUDA_COMPRESS_LEVEL, qs)
        compressed_size = len(compressed)
        stats['num_chunks'] = num_chunks
        stats['compressed_size_bytes'] = compressed_size
        stats['stats_compression'] = stats_compression

        img_bricked, stats_decompression = br.render(
            compressed, camera, width, height,
            stepsize_world=stepsize_world, timer=timer)
        time_bricked = timer.elapsed_milliseconds()
        stats['stats_decompression'] = stats_decompression
        stats['decompression_memory_bytes'] = \
            stats_decompression['peak_memory_cpu'] + stats_decompression['peak_memory_gpu']

        image_filename = '%s.png' % filename
        imageio.imwrite(
            os.path.join(image_folder, image_filename),
            LoadedModel.convert_image(img_bricked))

        stats.update({
            'image_filename': image_filename,
            'time_world_ms': float("nan"),
            'time_screen_ms': time_bricked,
            'psnr_world': float("nan"),
            'ssim_screen': float("nan"),
            'lpips_screen': float("nan")
        })

        with open(stat_filename, 'w') as f:
            json.dump(stats, f)
        return stats

    if config.use_chunks:
        for quantStep in cudaQuantX:
            case_stats.append(_evalCudaCompressBricked(quantStep))

    # save statistics
    print("\n===================================== Done, save statistics")
    output_stats['cases'] = case_stats
    with open(statistics_file, "w") as f:
        json.dump(output_stats, f)
    return statistics_file

def _plot(configX: List[Config], files: List[str], name_suffix:str):
    print("Make Plots")
    statsX = []
    for sf in files:
        with open(sf, 'r') as f:
            statsX.append(json.load(f))
    num_configs = len(configX)

    FIELDS = [
        # (Key, Human name, axis-x, optional major ticks)
        ("decompression_memory_bytes", "a) Peak Memory", "log", [(1024*16, "16KB"), (1024**2, "1MB"), (1024**3, "1GB"), (1024**3*16, "16GB")]),
        ("psnr_world", "b) PSNR, $10^{%d}$ points"%np.log10(num_points), "linear", None),
        ("time_world_ms", "c) Time (sec), $10^{%d}$ points" % np.log10(num_points), "log", [(1, '1ms'), (10, '10ms'), (100, '100ms'), (1000, '1s'), (1000*60, '1min')]),
        ("ssim_screen", "d) SSIM, $1024^2$ image", "linear", None),
        ("time_screen_ms", "e) Time (sec), $1024^2$ image", "log", [(1000, '1s'), (1000*60, '1min'), (1000*60*10, '10min')]),
    ]
    CLASSES = [
        # key, human name, color
        ("fv-SRN", "fV-SRN", "red"),
        ("neurcomp", "neurcomp", "blue"),
        ("TThresh", "TThresh", "darkgreen"),
        ("TThreshBricked", "TThresh - bricked", "limegreen"),
        ("cudaCompress", "cudaCompress", "black"),
        ("cudaCompressBricked", "cudaCompress - bricked", "gray"),
    ]
    CLASSES_IMAGES = ["fv-SRN", "TThresh", "cudaCompress"]
    classKeys = [c[0] for c in CLASSES]

    # collect the data
    datapointsX = [defaultdict(list) for _ in range(num_configs)]
    for j in range(num_configs):
        original_size_bytes = statsX[j]['original_size_bytes']
        for entry in statsX[j]['cases']:
            alg_index = classKeys.index(entry['algorithm'])
            compression_rate = original_size_bytes / entry['compressed_size_bytes'] # y-axis
            for i,f in enumerate(FIELDS):
                v = entry[f[0]]
                datapointsX[j][(alg_index, i)].append((compression_rate, v))
        # and sort for compression rate
        for vx in datapointsX[j].values():
            vx.sort(key=lambda cr_v:cr_v[0])

    def formatCompressionRate(y:float, _=None):
        if y<0: return "%d:1"%int(np.round(1/y))
        return "1:%d"%int(np.round(y))

    # print PSNR
    test_field = 1
    for j in range(num_configs):
        print(configX[j].name)
        for i,(c, c_human, c_color) in enumerate(CLASSES):
            print(c,FIELDS[test_field][1])
            vx = datapointsX[j][(i, 1)]
            for cr,v in vx:
                print(f"  {formatCompressionRate(cr, None)} -> {v}")

    fig, axes = plt.subplots(nrows=num_configs, ncols=len(FIELDS), sharex='col', sharey='row', figsize=(3*len(FIELDS), 2.5*num_configs))
    for cfg_idx in range(num_configs):
        for i, f in enumerate(FIELDS):
            ax = axes[cfg_idx,i]
            #plot
            for j, (c, c_human, c_color) in enumerate(CLASSES):
                data = datapointsX[cfg_idx][(j, i)]
                if len(data)>0:
                    X = [v for cr,v in data]
                    Y = [cr for cr,v in data]
                    ax.plot(X, Y, 'o-', label=c_human, color=c_color)
            #adjust axis
            ax.set_xscale(f[2])
            ax.set_yscale('log')
            if cfg_idx == num_configs-1:
                ax.set_xlabel(f[1]) #label
            if f[3] is not None:
                ax.set_xticks([tp for tp, tn in f[3]], minor=False)
                if cfg_idx == num_configs-1:
                    ax.set_xticklabels([tn for tp, tn in f[3]], minor=False)
                ## manually make minor ticks
                #left, right = axes[i].get_xlim()
                #loc = matplotlib.ticker.LogLocator(subs='all')
                #ticks = loc.tick_values(left, right)
                #axes[i].set_xticks(ticks, minor=True)
            ax.grid(visible=True, which='both', color='lightgray')
        axes[cfg_idx,0].set_ylabel(configX[cfg_idx].human_name + " - Compression") #("Compression")
        axes[cfg_idx,0].yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(formatCompressionRate))

    handles, labels = axes[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(CLASSES))
    #fig.suptitle(config.human_name)

    fig.align_ylabels(axes[:, 0])
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.08, top=0.94)
    output_filename = 'CompressionExtended%s.pdf'%name_suffix
    fig.savefig(os.path.join(BASE_PATH, output_filename), bbox_inches='tight')
    plt.close()
    print("Plot saved to", output_filename)


    # WRITE Images
    latex_filename = os.path.join(BASE_PATH, 'CompressionExtended%s-Images.tex'%name_suffix)
    images_folder = "images_selected"
    num_columns = (len(CLASSES_IMAGES)+1)*num_configs
    with open(latex_filename, "w") as latex:
        latex.write("""
\\documentclass[10pt,a4paper]{standalone}
\\usepackage{graphicx}
\\usepackage{multirow}
\\begin{document}
\\newcommand{\\imgsize}{0.25}%
\setlength{\\tabcolsep}{1pt}%
""")
        latex.write("\\begin{tabular}{%s}%%\n"%("c"*(num_columns)))
        # find closest match
        def dist(target, actual):
            #return np.abs(target-actual)
            return np.abs(np.log10(target) - np.log10(actual))
        matches_per_config = []
        for j in range(num_configs):
            target_rate = configX[j].target_compression_for_vis
            matches = [None] * len(CLASSES_IMAGES)
            original_size_bytes = statsX[j]['original_size_bytes']
            for i,cls in enumerate(CLASSES_IMAGES):
                best_distance = 10**10
                for entry in statsX[j]['cases']:
                    if entry['algorithm']!=cls: continue
                    compression_rate = original_size_bytes / entry['compressed_size_bytes']  # y-axis
                    distance = dist(target_rate, compression_rate)
                    if distance < best_distance:
                        best_distance = distance
                        matches[i] = (compression_rate, entry['ssim_screen'], entry['image_filename'])
            matches_per_config.append(matches)

        # write images
        def write_image(src_folder, name, sep_column=False):
            # copy to image folder
            s = os.path.join(BASE_PATH, "images_"+src_folder, name)
            d = os.path.join(BASE_PATH, images_folder, src_folder+"-"+name)
            shutil.copyfile(s,d)
            print("Copy", s, "to", d)
            # add to latex
            if sep_column:
                latex.write(" &")
            else:
                latex.write("  ")
            latex.write("\\includegraphics[width=\\imgsize\\linewidth]{%s/%s_lens}%%\n"%(
                images_folder, src_folder+"-"+os.path.splitext(name)[0]))
        for j in range(num_configs):
            # reference image
            write_image(configX[j].name, "reference.png", j>0)
            # case studies
            for i, cls in enumerate(CLASSES_IMAGES):
                write_image(configX[j].name, matches_per_config[j][i][2], True)
        latex.write("\\\\\n")

        # write statistics
        def num_to_char(i):
            return chr(ord('a')+i)+")~"
        for j in range(num_configs):
            # reference label
            if j>0: latex.write(" &")
            latex.write(num_to_char(j*(len(CLASSES_IMAGES)+1)))
            # case studies
            for i, cls in enumerate(CLASSES_IMAGES):
                rate, ssim, _ = matches_per_config[j][i]
                label = num_to_char(j*(len(CLASSES_IMAGES)+1) + i+1)
                latex.write(" &")
                latex.write("\\begin{tabular}{crl}")
                latex.write("\\multirow{2}{*}{%s}"%label)
                latex.write("&{\\footnotesize Compression}&{\\footnotesize %s}\\\\"%formatCompressionRate(rate))
                latex.write("&{\\footnotesize SSIM}&{\\footnotesize %.2f}" % ssim)
                latex.write("\\end{tabular}\n")

        latex.write("\\end{tabular}\n")
        latex.write("\\end{document}\n")

if __name__ == '__main__':
    #write_scripts_for_training()
    eval_and_plot()
