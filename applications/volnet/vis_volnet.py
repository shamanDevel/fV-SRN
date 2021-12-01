import os
import sys

sys.path.append(os.getcwd())

import h5py
import common.vis_gui
import torch
import numpy as np
import skimage.transform
import time
import enum
import re

from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import common.utils as utils
import pyrenderer

from volnet.network import SceneRepresentationNetwork
from volnet.input_data import TrainingInputData
from volnet.raytracing import Raytracing
from volnet.inference import LoadedModel

class RenderEngine(enum.Enum):
    PyTorch = enum.auto()
    TensorCores = enum.auto()


class UIVolnet(common.vis_gui.UI):
    STEPSIZE = 0.01

    def __init__(self, folder, filter=".*"):
        KEYS = [
            "filename", "timesteps", "ensembles",
            "mode", "output", "layers", "activation",
            "fourier",
            "lr",
        ]
        LOSSES = [
            "l1", "l2",
        ]

        self.number_of_steps = 10
        self.render_engine = RenderEngine.PyTorch
        self.filename_filer = re.compile(filter)

        super().__init__(
            folder,
            KEYS,
            LOSSES,
            512, 256,
            ["filename", "weights", "inference"],
            delayed_loading=True,
            has_volume_slices=False,
            allows_free_camera=True)
        self.folder = folder

    def _createKey(self, hdf5_file: h5py.File):
        if self.filename_filer.fullmatch(hdf5_file.filename) is None:
            raise ValueError("file %s is ignored, does not match filename filter"%os.path.split(hdf5_file.filename)[1])
        fourier = "%d:%.3f"%(
            hdf5_file.attrs["fouriercount"] if "fouriercount" in hdf5_file.attrs else 0,
            hdf5_file.attrs["fourierstd"] if "fourierstd" in hdf5_file.attrs else 1
        )
        layers = hdf5_file.attrs["layers"].split(':')
        return self.Key(
            filename = os.path.splitext(os.path.split(hdf5_file.filename)[1])[0],
            timesteps = hdf5_file.attrs["time_train"],
            ensembles = hdf5_file.attrs["ensembles"],
            mode="%s" % hdf5_file.attrs["train:mode"],
            output="%s" % hdf5_file.attrs["outputmode"],
            layers="%s^%d" % (layers[0], len(layers)),
            activation="%s" % hdf5_file.attrs["activation"],
            fourier=fourier,
            lr="%s" % hdf5_file.attrs["lr"],
        )

    def _createValue(self, hdf5_file: h5py.File, filename: str):
        return self.Value(
            filename = os.path.splitext(filename)[0],
            l1 = hdf5_file['l1'][...] if 'l1rgb' in hdf5_file else hdf5_file['l1'][...],
            l2 = hdf5_file['l2'][...] if 'l2rgb' in hdf5_file else hdf5_file['l2'][...],
            weights = hdf5_file['weights'], # lazy-loading
            inference = LoadedModel(hdf5_file),
        )

    def _load_root_module(self, current_entry) -> pyrenderer.IImageEvaluator:
        return current_entry.value.inference.get_image_evaluator()

    def prepare_rendering(self):
        super().prepare_rendering()
        # update timestep bounds
        inference = self.current_entry.value.inference
        input_data = inference.get_input_data()
        self._timestep_slider.setMinimum(0)
        self._timestep_slider.setMaximum(input_data.num_timesteps('val')-1)
        self._ensemble_slider.setMinimum(0)
        self._ensemble_slider.setMaximum(input_data.num_ensembles()-1)
        self._tf_slider.setMinimum(0)
        self._tf_slider.setMaximum(input_data.num_tfs() - 1)


    def render_reference(self, current_entry):
        inference = current_entry.value.inference
        camera = current_entry.root_module.camera.get_parameters()
        img = inference.render_reference(
            camera, self.ImgRes, self.ImgRes,
            self._selected_tf(), self._selected_timestep(), self._selected_ensemble())
        return utils.toHWC(img).cpu().numpy()[0]

    def _selected_timestep(self):
        return self._timestep_slider.value()
    def _selected_ensemble(self):
        return self._ensemble_slider.value()
    def _selected_tf(self):
        return self._tf_slider.value()

    def _render_engine_changed(self):
        engine = None
        for e,b in self._render_engine_buttons.items():
            if b.isChecked():
                engine = e
                break
        assert engine is not None
        print("Changed render engine to", engine)
        self.render_engine = engine
        self.epoch_slider_changed()

    def _num_steps_changed(self):
        self._num_steps_label.setText("%d"%self._num_steps_slider.value())
        self.epoch_slider_changed()

    def _export_current_network(self):
        if self.current_entry.value is None: return
        preferredFilename = self.current_entry.value.filename + "_epoch%03d.volnet" % self.selected_epoch
        filename = QFileDialog.getSaveFileName(
            self.window, "Export network",
            os.path.join(self.save_folder, preferredFilename),
            "VOLNET (*.volnet)")[0]
        if filename is not None and len(filename) > 0:
            self.current_entry.value.inference.save_compiled_network(filename)
            print("Saved to", filename)

    def time_or_ensemble_changed(self):
        self._update_time_ensemble_labels()

    def _update_time_ensemble_labels(self):
        if self.current_entry.value is None: return
        inference = self.current_entry.value.inference
        actual_timestep, actual_ensemble = inference.get_input_data().compute_actual_time_and_ensemble(
            self._selected_timestep(), self._selected_ensemble(), 'val')
        self._timestep_label.setText("Timestep: %3d" % actual_timestep)
        self._ensemble_label.setText("Ensemble: %3d" % actual_ensemble)
        self._tf_label.setText("TF: %2d" % self._selected_tf())

    def _custom_image_controls(self, parentLayout, parentWidget):

        # Row 1
        layout = QHBoxLayout(parentWidget)

        self._timestep_label = QLabel("Timestep:  0")
        layout.addWidget(self._timestep_label)
        self._timestep_slider = QSlider(Qt.Horizontal, parentWidget)
        self._timestep_slider.setTracking(True)
        self._timestep_slider.valueChanged.connect(self.time_or_ensemble_changed)
        layout.addWidget(self._timestep_slider)

        self._ensemble_label = QLabel("Ensemble:  0")
        layout.addWidget(self._ensemble_label)
        self._ensemble_slider = QSlider(Qt.Horizontal, parentWidget)
        self._ensemble_slider.setTracking(True)
        self._ensemble_slider.valueChanged.connect(self.time_or_ensemble_changed)
        layout.addWidget(self._ensemble_slider)

        self._tf_label = QLabel("TF:  0")
        layout.addWidget(self._tf_label)
        self._tf_slider = QSlider(Qt.Horizontal, parentWidget)
        self._tf_slider.setTracking(True)
        self._tf_slider.valueChanged.connect(self.time_or_ensemble_changed)
        layout.addWidget(self._tf_slider)
        parentLayout.addLayout(layout)

        # Row 2

        layout = QHBoxLayout(parentWidget)

        self._render_engine_button_group = QButtonGroup(parentWidget)
        layout.addWidget(QLabel("Render Engine:"))
        self._render_engine_buttons = {}
        for engine in RenderEngine:
            button = QRadioButton(engine.name, parentWidget)
            self._render_engine_buttons[engine] = button
            if self.render_engine == engine:
                button.setChecked(True)
            button.clicked.connect(lambda: self._render_engine_changed())
            layout.addWidget(button)

        self._export_network_button = QPushButton("export", parentWidget)
        self._export_network_button.clicked.connect(lambda: self._export_current_network())
        layout.addWidget(self._export_network_button)

        layout.addWidget(QLabel(" Num Steps:"))
        self._num_steps_slider = QSlider(Qt.Horizontal, parentWidget)
        self._num_steps_slider.setMinimum(5)
        self._num_steps_slider.setMaximum(100)
        self._num_steps_slider.setTracking(True)
        self._num_steps_slider.valueChanged.connect(self._num_steps_changed)
        layout.addWidget(self._num_steps_slider)
        self._num_steps_label = QLabel("5", parentWidget)
        layout.addWidget(self._num_steps_label)
        self._rendering_time_label = QLabel(" Time: xxx", parentWidget)
        layout.addWidget(self._rendering_time_label)

        parentLayout.addLayout(layout)


    def get_num_epochs(self, current_entry):
        return current_entry.value.weights.shape[0]

    #def get_transfer_function(self, current_value, current_epoch):
    #    return self.tf_reference

    def _update_network(self, current_entry, current_epoch):
        current_entry.value.inference.fill_weights(current_entry.value.weights, current_epoch)
        print("Network updated")

    def on_epoch_changed(self, current_value, current_epoch):
        self._update_network(current_value, current_epoch)

    def render_current_value(self, current_entry, current_epoch):
        engine = self.render_engine
        num_steps = self._num_steps_slider.value()
        stepsize = 1.0 / num_steps
        print("Render with engine", engine, "and a stepsize of", stepsize)

        time_start = time_end = 0

        inference = current_entry.value.inference
        assert isinstance(inference, LoadedModel)
        camera = current_entry.root_module.camera.get_parameters()
        #print("Camera:\n",camera.cpu().numpy())

        pyrenderer.sync()
        time_start = time.time()
        if engine == RenderEngine.PyTorch:
            out = inference.render_network(camera, self.ImgRes, self.ImgRes, LoadedModel.EvaluationMode.PYTORCH16, stepsize) # TODO: ensemble+timestep
        elif engine == RenderEngine.TensorCores:
            if inference.is_tensorcores_available():
                out = inference.render_network(camera, self.ImgRes, self.ImgRes, LoadedModel.EvaluationMode.TENSORCORES_MIXED, stepsize)  # TODO: ensemble+timestep
            else:
                print("No tensorcore implementation possible")
                out = torch.zeros(1,4,2,2) # empty
        else:
            print("Unknown engine")
            return None
        out = out.movedim(1,3)

        pyrenderer.sync()
        time_end = time.time()
        time_ms = (time_end - time_start) * 1000.0
        self._rendering_time_label.setText(" Time: %dms"%int(time_ms))
        return out.detach().cpu().numpy()[0]

    def get_slice(self, is_reference: bool, current_value, current_epoch,
                  slice: float, axis : str):
        # TODO: fix it
        # create slice_data of shape (C,H,W)
        if is_reference:
            volume = self.volume.getDataCpu(0).numpy()
            if axis == 'x':
                slice_index = int(slice * (volume.shape[1] - 1))
                slice_data = volume[:, slice_index, :, :]
            elif axis == 'y':
                slice_index = int(slice * (volume.shape[2] - 1))
                slice_data = volume[:, :, slice_index, :]
            elif axis == 'z':
                slice_index = int(slice * (volume.shape[3] - 1))
                slice_data = volume[:, :, :, slice_index]
            else:
                raise ValueError("Unknown slice axis: " + axis)
        else:
            # get positions
            box_min = np.array([0,0,0], renderer_dtype_np)#cvector_to_numpy(self.renderer.settings.box_min)
            box_size = np.array([1,1,1], renderer_dtype_np)#cvector_to_numpy(self.renderer.settings.box_size)
            if axis == 'x':
                #grid = create_grid()
                box_min[0] = box_min[0] + box_size[0] * slice
                Y = torch.linspace(box_min[1], box_min[1] + box_size[1], self.ImgRes)
                Z = torch.linspace(box_min[2], box_min[2] + box_size[2], self.ImgRes)
                X = torch.tensor(box_min[0])
                grid = torch.stack(torch.meshgrid(X, Y, Z), 3).to(device=self.device)
            elif axis == 'y':
                box_min[1] = box_min[1] + box_size[1] * slice
                X = torch.linspace(box_min[0], box_min[0] + box_size[0], self.ImgRes)
                Z = torch.linspace(box_min[2], box_min[2] + box_size[2], self.ImgRes)
                Y = torch.tensor(box_min[1])
                grid = torch.stack(torch.meshgrid(X, Y, Z), 3).to(device=self.device)
            elif axis == 'z':
                box_min[2] = box_min[2] + box_size[2] * slice
                X = torch.linspace(box_min[0], box_min[0] + box_size[0], self.ImgRes)
                Y = torch.linspace(box_min[1], box_min[1] + box_size[1], self.ImgRes)
                Z = torch.tensor(box_min[2])
                grid = torch.stack(torch.meshgrid(X, Y, Z), 3).to(device=self.device)
            else:
                raise ValueError("Unknown slice axis: " + axis)

            # evaluate network
            with torch.no_grad():
                full_network = current_value.network_loader.full_network
                network_input = grid.view((self.ImgRes*self.ImgRes, 3))
                network_output = full_network(network_input)
            assert len(network_output.shape)==2
            assert network_output.shape[0] == self.ImgRes*self.ImgRes
            C = network_output.shape[1]

            # transform to C*H*W
            network_output_hwc = network_output.view((self.ImgRes, self.ImgRes, C))
            slice_data = np.transpose(network_output_hwc.detach().cpu().numpy(), (2,0,1))

        if slice_data.shape[0] == 1:
            print("Slice-Density min:", np.min(slice_data), ", max:", np.max(slice_data),
                  "of reference" if is_reference else "")
            slice_rgb = np.stack([slice_data[0]] * 3, axis=2)
        else:
            print("Slice-Opacity min:", np.min(slice_data[3]), ", max:", np.max(slice_data[3]),
                  "of reference" if is_reference else "")
            slice_rgb = np.stack([slice_data[0], slice_data[1], slice_data[2]], axis=2)
            slice_rgb = slice_rgb * np.clip(slice_data[3,:,:,np.newaxis], None, 1) # opacity
        slice_rgb = skimage.transform.resize(slice_rgb, (self.ImgRes, self.ImgRes))
        return slice_rgb


if __name__ == "__main__":

    result_folder = os.path.join(os.path.split(__file__)[0], "results")

    ui = UIVolnet(os.path.join(result_folder, "eval_network_configs", "hdf5"), '.*SnakeAlt.*')

    ui.show()
