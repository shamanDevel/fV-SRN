"""
From a pre-trained model on certain timesteps and ensembles,
re-train for a new ensemble by re-learning the ensemble grid
"""

import sys
import os
sys.path.insert(0, os.getcwd())

import numpy as np
import torch
import torch.nn.functional as F
import os
import tqdm
import time
import h5py
import argparse
import shutil
import subprocess
import io
from contextlib import ExitStack
from collections import defaultdict, OrderedDict
import imageio
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid
from PIL import Image, ImageDraw, ImageFont

import common.utils as utils
import pyrenderer

from volnet.network import SceneRepresentationNetwork
from volnet.lossnet import LossFactory
from volnet.input_data import TrainingInputData
from volnet.training_data import TrainingData
from volnet.optimizer import Optimizer
from volnet.evaluation import EvaluateWorld, EvaluateScreen

def main():
    # Settings
    parser = argparse.ArgumentParser(
        description='Scene representation networks',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    def _extra_input_args(g):
        g.add_argument('trained_network', type=str,
                       help=".hdf5 file with the pre-trained network.")
        g.add_argument('--trained_network_epoch', type=int, default=-1,
            help="The checkpoint to use for loading the weights")
    TrainingInputData.init_parser(parser, _extra_input_args)
    TrainingData.init_parser(parser)
    #SceneRepresentationNetwork.init_parser(parser) #loaded from checkpoint
    LossFactory.init_parser(parser)
    Optimizer.init_parser(parser)

    parser_group = parser.add_argument_group("Output")
    this_folder = os.path.split(__file__)[0]
    parser_group.add_argument('--logdir', type=str, default=os.path.join(this_folder, 'results/log'),
                              help='directory for tensorboard logs')
    parser_group.add_argument('--modeldir', type=str, default=os.path.join(this_folder, 'results/model'),
                              help='Output directory for the checkpoints')
    parser_group.add_argument('--hdf5dir', type=str, default=os.path.join(this_folder, 'results/hdf5'),
                              help='Output directory for the hdf5 summary files')
    parser_group.add_argument('--name', type=str, default=None,
                              help='Output name. If not specified, use the next available index')
    parser_group.add_argument('--save_frequency', type=int, default=10,
                              help='Every that many epochs, a checkpoint is saved')
    parser_group.add_argument('--profile', action='store_true')
    parser.add_argument('--seed', type=int, default=124, help='random seed to use. Default=124')

    opt = vars(parser.parse_args())
    torch.manual_seed(opt['seed'])
    np.random.seed(opt['seed'])
    torch.set_num_threads(4)
    #torch.backends.cudnn.benchmark = True

    dtype = torch.float32
    device = torch.device("cuda")
    profile = opt['profile']

    # LOAD / initialize

    # Load the pre-trained model
    print("Open pre-trained model from", opt['trained_network'])
    with h5py.File(opt['trained_network'], 'r') as f:
        pretrained_opt = defaultdict(lambda: None)
        pretrained_opt.update(f.attrs)
        weights_np = f['weights'][opt['trained_network_epoch'], :]
        weights_bytes = io.BytesIO(weights_np.tobytes())

    # input data
    # Important: the time keyframes must match!!
    _num_input_keyframes = len(range(*map(int, opt['time_keyframes'].split(':'))))
    if _num_input_keyframes>1 and opt['time_keyframes'] != pretrained_opt['time_keyframes']:
        print("ERROR: For generalization, the time keyframes must match!\nFrom the network: ",
              pretrained_opt['time_keyframes'], ", from the command line:", opt['time_keyframes'])
        exit(-1)
    input_data = TrainingInputData(opt)

    # network
    # for loading, I need the original number of keyframes and ensembles
    class FakeTrainingInputData(TrainingInputData):
        # noinspection PyMissingConstructor
        def __init__(self):
            # super().__init__() deliberatly not called
            self._num_timekeyframes = len(range(*map(int, pretrained_opt['time_keyframes'].split(':'))))
            self._num_ensembles = len(range(*map(int, pretrained_opt['ensembles'].split(':'))))
        def num_timekeyframes(self):
            return self._num_timekeyframes
        def num_ensembles(self):
            return self._num_ensembles
    network = SceneRepresentationNetwork(pretrained_opt, FakeTrainingInputData(), dtype, device)
    network.load_state_dict(
        torch.load(weights_bytes, map_location=device), strict=True)
    data_to_train = network.generalize_to_new_ensembles(input_data.num_ensembles())
    network_output_mode = network.output_mode()
    network.to(device, dtype)

    # dataloader
    print("Create the dataloader")
    training_data = TrainingData(opt, dtype, device)
    training_data.create_dataset(input_data, network_output_mode, network.supports_mixed_latent_spaces())
    if network.use_direction() and (
            training_data.training_mode() == 'world' or training_data.validation_mode() == 'world'):
        print(
            "ERROR: The network requires the direction as input, but world-space training or validation was requested.")
        print(" Directions are only available for pure screen-space training")
        exit(-1)

    # losses
    loss_screen, loss_world, loss_world_mode = LossFactory.createLosses(opt, dtype, device)
    loss_screen.to(device, dtype)
    loss_world.to(device, dtype)

    # optimizer
    optimizer = Optimizer(opt, [data_to_train], dtype, device)

    # evaluation helpers
    if training_data.training_mode() == 'world':
        evaluator_train = EvaluateWorld(
            network, input_data.default_image_evaluator(), loss_world, dtype, device)
    else:
        evaluator_train = EvaluateScreen(
            network, input_data.default_image_evaluator(), loss_screen,
            training_data.training_image_size(), training_data.training_image_size(),
            True, training_data.train_disable_inversion_trick(), dtype, device)
    if training_data.validation_mode() == 'world':
        evaluator_val = EvaluateWorld(
            network, input_data.default_image_evaluator(), loss_world, dtype, device)
    else:
        evaluator_val = EvaluateScreen(
            network, input_data.default_image_evaluator(), loss_screen,
            training_data.validation_image_size(), training_data.validation_image_size(),
            False, False, dtype, device)
    evaluator_vis = EvaluateScreen(
        network, input_data.default_image_evaluator(), loss_screen,
        training_data.visualization_image_size(), training_data.visualization_image_size(),
        False, False, dtype, device)
    try:
        vis_fnt = ImageFont.truetype("arial.ttf", 12)
    except OSError:
        # Unix, try free-font
        try:
            vis_fnt = ImageFont.truetype("FreeSans.ttf", 12)
        except OSError:
            vis_fnt = ImageDraw.getfont()  # fallback

    # Create the output
    print("Model directory:", opt['modeldir'])
    print("Log directory:", opt['logdir'])
    print("HDF5 directory:", opt['hdf5dir'])
    # update opt-dictionary with missing keys from pretrained_opt
    # so that loading the new network without the pretrained one works
    opt = {**pretrained_opt, **opt} # first pretrained, then normal ops, as the latter overwrites the former

    def findNextRunNumber(folder):
        if not os.path.exists(folder): return 0
        files = os.listdir(folder)
        files = sorted([f for f in files if f.startswith('run')])
        if len(files) == 0:
            return 0
        return int(files[-1][3:])

    overwrite_output = False
    if opt['name'] == None:
        nextRunNumber = max(findNextRunNumber(opt['logdir']), findNextRunNumber(opt['modeldir'])) + 1
        print('Current run: %05d' % nextRunNumber)
        runName = 'run%05d' % nextRunNumber
    else:
        runName = opt['name']
        overwrite_output = True
    logdir = os.path.join(opt['logdir'], runName)
    modeldir = os.path.join(opt['modeldir'], runName)
    hdf5file = os.path.join(opt['hdf5dir'], runName + ".hdf5")
    if overwrite_output and (os.path.exists(logdir) or os.path.exists(modeldir) or os.path.exists(hdf5file)):
        print(f"Warning: Overwriting previous run with name {runName}")
        if os.path.exists(logdir):
            shutil.rmtree(logdir)
    os.makedirs(logdir, exist_ok=overwrite_output)
    os.makedirs(modeldir, exist_ok=overwrite_output)
    os.makedirs(opt['hdf5dir'], exist_ok=True)

    optStr = str(opt)
    print(optStr)
    with open(os.path.join(modeldir, 'info.txt'), "w") as text_file:
        text_file.write(optStr)
    with open(os.path.join(modeldir, 'cmd.txt'), "w") as text_file:
        import shlex
        text_file.write('cd "%s"\n' % os.getcwd())
        text_file.write(' '.join(shlex.quote(x) for x in sys.argv) + "\n")

    # tensorboard logger
    writer = SummaryWriter(logdir)
    writer.add_text('info', optStr, 0)

    # compute epochs
    epochs = optimizer.num_epochs() + 1
    epochs_with_save = set(list(range(0, epochs - 1, opt['save_frequency'])) + [-1, epochs - 1])

    # HDF5-output for summaries and export
    with h5py.File(hdf5file, 'w') as hdf5_file:
        for k, v in opt.items():
            try:
                hdf5_file.attrs[k] = v
            except TypeError as ex:
                print("Exception", ex, "while saving attribute", k, "=", str(v))
        try:
            git_commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
            hdf5_file.attrs['git'] = git_commit
            print("git commit", git_commit)
        except:
            print("unable to get git commit")

        times = hdf5_file.create_dataset("times", (epochs,), dtype=np.float32)
        losses = dict([
            (name, hdf5_file.create_dataset(name, (epochs,), dtype=np.float32))
            for name in evaluator_val._loss.loss_names()
        ])

        def save_network(net):
            weights_bytes = io.BytesIO()
            torch.save(net.state_dict(), weights_bytes)
            return np.void(weights_bytes.getbuffer())

        weights = hdf5_file.create_dataset(
            "weights",
            (len(epochs_with_save), save_network(network).shape[0]),
            dtype=np.dtype('V1'))
        export_weights_counter = 0

        def trace_handler(prof):
            print(prof.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=-1))
            prof.export_chrome_trace("test_trace_" + str(prof.step_num) + ".json")

        start_time = time.time()
        with ExitStack() as stack:
            iteration_bar = stack.enter_context(tqdm.tqdm(total=epochs))
            if profile:
                profiler = stack.enter_context(torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ], schedule=torch.profiler.schedule(
                        wait=1,
                        warmup=1,
                        active=2),
                    on_trace_ready=trace_handler))
            for epoch in range(-1, epochs):
                # special case epoch==-1 -> only visualize
                if epoch >= 0:
                    # update network
                    if network.start_epoch():
                        optimizer.reset(network.parameters())
                    # update training data
                    if training_data.is_rebuild_dataset():
                        if (epoch + 1) % training_data.rebuild_dataset_epoch_frequency() == 0:
                            training_data.rebuild_dataset(
                                input_data, network_output_mode, network)
                    # TRAIN
                    partial_losses = defaultdict(float)
                    network.train()
                    num_batches = 0
                    for data_tuple in training_data.training_dataloader():
                        num_batches += 1
                        data_tuple = utils.toDevice(data_tuple, device)

                        def optim_closure():
                            nonlocal partial_losses
                            optimizer.zero_grad()
                            prediction, total, lx = evaluator_train(data_tuple)
                            for k, v in lx.items():
                                partial_losses[k] += v
                            total.backward()
                            # print("Grad latent:", torch.sum(network._time_latent_space.grad.detach()).item())
                            # print("Batch, loss:", total.item())
                            return total

                        optimizer.step(optim_closure)

                    for k, v in partial_losses.items():
                        writer.add_scalar('train/%s' % k, v / num_batches, epoch)
                    writer.add_scalar('train/lr', optimizer.get_lr()[0], epoch)
                    # print("Training epoch done, total:", partial_losses['total']/num_batches)

                # save checkpoint
                if epoch in epochs_with_save:
                    # save to tensorboard
                    model_out_path = os.path.join(modeldir, "model_epoch_{}.pth".format(epoch if epoch>=0 else "init"))
                    state = {'epoch': epoch + 1, 'model': network, 'parameters': opt}
                    torch.save(state, model_out_path)
                    print("Checkpoint saved to {}".format(model_out_path))
                    # save to HDF5-file
                    weights[export_weights_counter, :] = save_network(network)
                    export_weights_counter += 1

                # VALIDATE
                if epoch>=0:
                    partial_losses = defaultdict(float)
                    network.eval()
                    num_batches = 0
                    with torch.no_grad():
                        for j, data_tuple in enumerate(training_data.validation_dataloader()):
                            num_batches += 1
                            data_tuple = utils.toDevice(data_tuple, device)
                            prediction, total, lx = evaluator_val(data_tuple)
                            for k, v in lx.items():
                                partial_losses[k] += v

                        for k, v in partial_losses.items():
                            writer.add_scalar('val/%s' % k, v / num_batches, epoch)
                        end_time = time.time()
                        times[epoch] = end_time - start_time
                        for k, v in partial_losses.items():
                            losses[k][epoch] = v / num_batches

                # VISUALIZE
                if epoch in epochs_with_save:
                    with torch.no_grad():
                        # the vis dataset contains one entry per tf-timestep-ensemble
                        # -> concatenate them into one big image
                        left = 50
                        bottom = 100
                        text_fill = (120, 120, 120, 255)
                        img_size = training_data.visualization_image_size()
                        num_ensembles = input_data.num_ensembles()
                        num_tfs = input_data.num_tfs()
                        width = left + (len(training_data.visualization_dataloader()) // num_ensembles) * img_size
                        single_height = bottom + 2 * img_size
                        height = single_height * num_ensembles
                        image = Image.new('RGBA', (width, height))
                        draw = ImageDraw.Draw(image)
                        draw.text((5, img_size // 2), "pred", fill=text_fill, font=vis_fnt)
                        draw.text((5, img_size + img_size // 2), "gt", fill=text_fill, font=vis_fnt)
                        for j, data_tuple in enumerate(training_data.visualization_dataloader()):
                            target = data_tuple[1]
                            tf_index = data_tuple[2].item()
                            time_index = data_tuple[3].item()
                            ensemble_index = data_tuple[4].item()
                            data_tuple = utils.toDevice(data_tuple, device)
                            prediction, total, lx = evaluator_vis(data_tuple)

                            posX = j // num_ensembles
                            posY = j % num_ensembles

                            def convert_image(img):
                                out_img = img[0].cpu().detach().numpy()
                                out_img *= 255.0
                                out_img = out_img.clip(0, 255)
                                out_img = np.uint8(out_img)
                                out_img = np.moveaxis(out_img, (1, 2, 0), (0, 1, 2))
                                return Image.fromarray(out_img)

                            image.paste(convert_image(prediction),
                                        box=(left + posX * img_size, posY * single_height))
                            image.paste(convert_image(target),
                                        box=(left + posX * img_size, posY * single_height + img_size))

                            def centered_text(text, x_left, y):
                                w, h = draw.textsize(text)
                                x = int(x_left + img_size / 2 - w / 2)
                                draw.text((x, y), text, fill=text_fill, font=vis_fnt)
                                return y + h

                            y = centered_text(
                                "TF=%d, Time=%.2f, Ensemble=%.2f" % (tf_index, time_index, ensemble_index),
                                left + posX * img_size, posY * single_height + 2 * img_size + 10)
                            y = centered_text("DSSIM: %.5f" % lx['dssim'], left + posX * img_size, y + 5)
                            y = centered_text("LPIPS: %.5f" % lx['lpips'], left + posX * img_size, y + 5)

                        # save image
                        del draw
                        img_np = np.array(image)
                        imageio.imwrite(os.path.join(logdir, ("e%d.png" % epoch) if epoch>=0 else "eInit.png"), img_np)
                        writer.add_image('vis', np.moveaxis(img_np, (0, 1, 2), (1, 2, 0)), epoch)

                # done with this epoch
                if profile:
                    profiler.step()
                iteration_bar.update(1)
                if epoch>=0:
                    optimizer.post_epoch()
                    final_loss = partial_losses['total'] / max(1, num_batches)
                    iteration_bar.set_description("Loss: %7.5f" % (final_loss))
                    if np.isnan(final_loss):
                        break

    print("Done in", (time.time() - start_time), "seconds")

if __name__ == '__main__':
    main()
