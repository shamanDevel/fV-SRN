import os
import sys

sys.path.append(os.getcwd())

import numpy as np
import torch
import os
import time
import h5py
import collections
import matplotlib
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

# import PyQtChart
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtChart import *

import common.utils as utils
import pyrenderer

class CommonVariables:
    run_on_cuda = True
    screen_height = 512
    screen_width = 512

class Entry:
    def __init__(self, value, settings_file, root_module=None):
        self.value = value
        self.settings_file = settings_file
        self.root_module = root_module
        self.pitch_yaw_distances = None

class UI(ABC):

    def __init__(
            self, folder,
            KeyNames, LossNames, ImgRes, TfHeight, ExtraValues = None,
            delayed_loading=False, has_tf=False, has_volume_slices=False, allows_free_camera=False,
            use_tf_from_settings=False):
        self.folder = folder
        self.save_folder = os.path.split(folder)[0]
        self.delayed_loading = delayed_loading
        self.has_tf = has_tf
        self.has_volume_slices = has_volume_slices
        self.allows_free_camera = allows_free_camera
        self.use_tf_from_settings = use_tf_from_settings

        self.device = torch.device("cuda")
        self.renderer_dtype_torch = torch.float32
        self.renderer_dtype_np = np.float32

        self.KeyNames = KeyNames
        self.Key = collections.namedtuple("Key", KeyNames)
        self.LossNames = LossNames
        if ExtraValues is None:
            ExtraValues = ["filename"]
        self.Value = collections.namedtuple("Value", LossNames + ExtraValues)
        self.ImgRes = ImgRes
        self.TFHeight = TfHeight
        self.ExportFileNames = "PNG (*.png);;JPEG (*.jpeg);;Bitmap (*.bmp)"

        self.settings_file = None
        self.tf_reference = None
        self.tf_reference_torch = None
        self.tf_mode = None
        self.vis_mode = "image" # image, tf, slice
        self.slice_axis = 'x' # x,y,z
        self.bar_series = dict()
        self.white_background = False
        self.selected_epoch = -1

        self.camera_follows_settings = True
        self.camera_custom_yaw = 0
        self.camera_custom_pitch = 0
        self.camera_disable_slider_events = False
        self.camera_selection = 0

        self.vis()

        self.reparse()

        self.img_reference_pixmap = None
        self.img_current_pixmap = None
        self.tf_reference_pixmap = None
        self.tf_current_pixmap = None
        self.slice_reference_pixmap = None
        self.slice_current_pixmap = None
        self.current_slice = 0
        self.lineseries_list = []

    def show(self):
        self.window.show()
        self.a.exec_()

    @abstractmethod
    def _createKey(self, hdf5_file: h5py.File):
        ...

    @abstractmethod
    def _createValue(self, hdf5_file: h5py.File, filename: str):
        ...

    def reparse(self):
        self.entries = self.parse(self.folder)
        self.entry_list = list(sorted(self.entries.items()))
        if len(self.entry_list)>0:
            self.prepare_colormaps()
        self.current_entry = None

        # fill data
        self.tableWidget.setRowCount(len(self.entries))
        for r, (k, v) in enumerate(self.entry_list):
            for c in range(len(self.KeyNames)):
                self.tableWidget.setItem(r, c, QTableWidgetItem(k[c]))
            for c,lossname in enumerate(self.LossNames):
                loss = getattr(v.value, lossname)
                value = loss[-1] if (loss is not None and len(loss) > 0) else -1
                item = QTableWidgetItem("%.5f" % value)
                if value <= 0:
                    color = [1,1,1]
                else:
                    color = self.colorbar(self.normalizations[c](np.log(value)))
                item.setBackground(QColor(
                    int(color[0] * 255), int(color[1] * 255), int(color[2] * 255)
                ))
                self.tableWidget.setItem(
                    r, c + len(self.KeyNames), item)

    def parse(self, folder):
        entries = dict()
        time_start = time.time()
        for filename in os.listdir(folder):
            if not filename.endswith(".hdf5"):
                continue
            hdf5_file = None
            try:
                hdf5_file = h5py.File(os.path.join(folder, filename), 'r')
                # get key
                if 'optimizer' in hdf5_file.attrs and hdf5_file.attrs['optimizer']=="Adamdelta":
                    continue
                key = self._createKey(hdf5_file)
                # get value
                value = self._createValue(hdf5_file, filename)
                # settings
                settings_file = hdf5_file.attrs['settingsFile'] if 'settingsFile' in hdf5_file.attrs else hdf5_file.attrs['settings']
                if not os.path.exists(settings_file):
                    print("Warning: settings file \"%s\" not found"%settings_file)
                    print("Hack: use c60-v1.json")
                    settings_file = "neuraltextures/config-files/c60-v1.json"
                #tf_mode = hdf5_file.attrs['tfmode'] if 'tfmode' in hdf5_file.attrs else 'texture'
                #tf_reference = hdf5_file["reference_tf"][...] if "reference_tf" in hdf5_file else None
                # add entry
                entries[key] = Entry(value, settings_file)
                if not self.delayed_loading:
                    hdf5_file.close()
                print("Loaded:", filename)
            except Exception as e:
                print("Unable to load file", filename, ":", e)
                if hdf5_file is not None:
                    hdf5_file.close()
        time_end = time.time()
        print("Folder parsed in", (time_end-time_start), "seconds with", len(entries), " entries")
        return entries

    def exportAsFile(self, file, sep='\t', newline='\n'):
        if isinstance(file, str):
            with open(file, "w") as f:
                self.exportAsFile(f, sep, newline)
                return

        # header
        file.write("ID"+sep)
        file.write(sep.join(self.KeyNames))
        file.write(sep+"Epoch"+sep)
        file.write(sep.join(self.LossNames))
        file.write(newline)

        # entries
        for r, (k, v) in enumerate(self.entry_list):
            epochs = len(v.value[0])
            for epoch in range(epochs):
                file.write("%d%s"%(r, sep))
                file.write(sep.join([str(k[c]) for c in range(len(self.KeyNames))]))
                file.write("%s%d%s"%(sep, epoch, sep))
                file.write(sep.join([
                    (("%.5f"%v.value[c][epoch]) if v.value[c] is not None else "") for c in range(len(self.LossNames))
                ]))
                file.write(newline)

    def _load_root_module(self, current_entry: Entry) -> pyrenderer.IImageEvaluator:
        return pyrenderer.load_from_json(current_entry.settings_file)

    def prepare_rendering(self):
        assert isinstance(self.current_entry, Entry)
        if self.current_entry.root_module is None:
            self.current_entry.root_module = self._load_root_module(self.current_entry)

            num_cameras = self.get_num_cameras(self.current_entry)
            self.current_entry.pitch_yaw_distances = [
                self.current_entry.root_module.camera.pitchYawDistance.value
            ]
            _camera_distance = self.current_entry.root_module.camera.pitchYawDistance.value.z
            if num_cameras>1:
                _camera_pitch_cpu, _camera_yaw_cpu = utils.fibonacci_sphere(num_cameras-1)
                for i in range(num_cameras-1):
                    self.current_entry.pitch_yaw_distances.append(pyrenderer.double3(
                        _camera_pitch_cpu[i], _camera_yaw_cpu[i], _camera_distance
                    ))
            print("cameras:", ",".join([str(c) for c in self.current_entry.pitch_yaw_distances]))

        self.selected_epoch = -1
        self._updateCamera(self.current_entry.pitch_yaw_distances[0], False, True)
        self.visualize_reference()

    def get_num_cameras(self, current_entry:Entry):
        """
        Returns the number of pre-defined cameras.
        By default: just the camera from the settings file
        """
        return 1

    def get_camera_settings(self, current_entry:Entry, id):
        return current_entry.pitch_yaw_distances[id]

    def _updateCamera(self, pitchYawDistance, redraw, update_sliders):
        if self.camera_follows_settings:
            pitchYawDistance = self.get_camera_settings(self.current_entry, self.camera_selection)
            if update_sliders:
                old_value = self.camera_disable_slider_events
                self.camera_disable_slider_events = True
                self.camera_yaw_slider.setValue(np.rad2deg(pitchYawDistance.y))
                self.camera_pitch_slider.setValue(np.rad2deg(pitchYawDistance.x))
                self.camera_disable_slider_events = old_value

        self.current_entry.root_module.camera.set_parameters(torch.Tensor()) # hack, reset camera so that we can use pitchYawDistance again
        self.current_entry.root_module.camera.pitchYawDistance.value = pitchYawDistance
        #print("Camera0:\n", self.current_entry.root_module.camera.get_parameters().cpu().numpy())

        if redraw:
            self.visualize_reference()
            #self.epoch_slider_changed()
            img = self.render_current_value(self.current_entry, self.selected_epoch)
            self.img_current_pixmap = self.to_pixmap(img)
            if self.vis_mode == 'image':
                self.current_label.setPixmap(self.img_current_pixmap)

    def render_reference(self, current_entry):
        img = current_entry.root_module.render(self.ImgRes, self.ImgRes)
        img = current_entry.root_module.extract_color(img)
        return utils.toHWC(img).cpu().numpy()[0]

    def to_pixmap(self, img : np.ndarray):
        h, w, c = img.shape
        if c>3:
            if self.white_background:
                rgb = img[:,:,:3]
                alpha = img[:,:,3:4]
                white = np.ones_like(rgb)
                img = alpha*rgb + (1-alpha)*white
            else:
                img = img[:,:,:3]
        img = np.ascontiguousarray(np.clip(255*img, 0, 255).astype(np.uint8))
        bytesPerLine = 3 * w
        qImg = QImage(img.data, w, h, bytesPerLine, QImage.Format_RGB888)
        return QPixmap(qImg)


    def prepare_colormaps(self):
        self.normalizations = []
        for i,lossname in enumerate(self.LossNames):
            data = [None]*len(self.entry_list)
            for j in range(len(self.entry_list)):
                loss = getattr(self.entry_list[j][1].value, lossname)
                if loss is not None and len(loss)>0:
                    data[j] = loss[-1]
                else:
                    data[j] = 1
            data = np.log(np.clip(data, 1e-5, None))
            self.normalizations.append(matplotlib.colors.Normalize(
                np.min(data), np.max(data)
            ))
            print("Loss", self.LossNames[i], "-> min:", np.min(data), ", max:", np.max(data))
        self.colorbar = plt.get_cmap("Reds").reversed()


    def visualize_reference(self):
        # render reference
        img_reference_data = self.render_reference(self.current_entry)
        self.img_reference_pixmap = self.to_pixmap(img_reference_data)
        self.reference_label.setPixmap(self.img_reference_pixmap)
        if self.has_tf:
            self.tf_reference_pixmap = self.visualize_tf(self.tf_reference, QPixmap(self.ImgRes, self.TFHeight))


    def visualize_tf(self, tf: np.ndarray, pixmap: QPixmap):
        _, R, _ = tf.shape
        W, H = pixmap.width(), pixmap.height()
        painter = QPainter(pixmap)

        def lerp(a, b, x):
            return (1 - x) * a + x * b

        if self.tf_mode == "texture":
            interpX = np.linspace(0, 1, W, endpoint=True)
            interpXp = np.array([(i+0.5)/R for i in range(R)])
            rx = np.interp(interpX, interpXp, tf[0, :, 0])
            gx = np.interp(interpX, interpXp, tf[0, :, 1])
            bx = np.interp(interpX, interpXp, tf[0, :, 2])
            ox = tf[0, :, 3]
            oxX = [0] + list(interpXp) + [1]
            oxY = [ox[0]] + list(ox) + [ox[-1]]
            max_opacity = np.max(ox)

            for x in range(W):
                painter.fillRect(x, 0, 1, H, QBrush(QColor(int(rx[x]*255), int(gx[x]*255), int(bx[x]*255))))

            lower = int(0.8 * H)
            upper = int(0.05 * W)
            def transform(x, y):
                return int(x*W), int(lerp(lower, upper, y/max_opacity))
            p1 = QPainterPath()
            p1.moveTo(0, 0)
            for x,y in zip(oxX, oxY):
                p1.lineTo(*transform(x, y))
            p1.lineTo(W, 0)
            p1.lineTo(0, 0)
            painter.fillPath(p1, QColor(255, 255, 255))

            pen = painter.pen()
            painter.setPen(QPen(QBrush(QColor(0,0,0)), 2, Qt.DashLine))
            painter.drawLine(0, lower, W, lower)

            painter.setPen(QPen(QBrush(QColor(0, 0, 0)), 5, Qt.SolidLine))
            points = [QPoint(*transform(x,y)) for (x,y) in zip(oxX, oxY)]
            points = [[p1, p2] for (p1, p2) in zip(points[:-1], points[1:])]
            points = [p for p2 in points for p in p2 ]
            painter.drawLines(*points)
            painter.setBrush(QColor(0,0,0))
            for x, y in zip(oxX[1:-1], oxY[1:-1]):
                ix, iy = transform(x, y)
                painter.drawEllipse(int(ix-2), int(iy-2), 4, 4)
            painter.setPen(pen)

        elif self.tf_mode == "linear":
            pass

        elif self.tf_mode == "gauss":
            painter.fillRect(0, 0, W, H, QColor(50, 50, 50))
            # compute gaussians
            X = np.linspace(0, 1, W, endpoint=True)
            R = tf.shape[1]
            def normal(x, mean, variance):
                return np.exp(-(x-mean)*(x-mean)/(2*variance*variance))
            Yx = [None] * R
            max_opacity = 0
            for r in range(R):
                red, green, blue, opacity, mean, variance = tuple(tf[0,r,:])
                Yx[r] = normal(X, mean, variance)*opacity
                max_opacity = max(max_opacity, opacity)
            # draw background gaussians
            for r in range(R):
                red, green, blue, _, _, _ = tuple(tf[0,r,:])
                col = QColor(int(red*255), int(green*255), int(blue*255), 100)
                for x in range(W):
                    y = int(Yx[r][x] * H / max_opacity)
                    if y>2:
                        painter.fillRect(x, H-y, 1, y, col)
            # draw foreground gaussians
            for r in range(R):
                red, green, blue, _, _, _ = tuple(tf[0,r,:])
                col = QColor(int(red*255), int(green*255), int(blue*255), 255)
                for x in range(W):
                    y = int(Yx[r][x] * H / max_opacity)
                    if y>2:
                        painter.fillRect(x, H-y, 1, 2, col)


            # draw foreground gaussians

        del painter
        return pixmap

    @abstractmethod
    def get_num_epochs(self, current_entry:Entry):
        ...

    def selection_changed(self, row, column):
        print("Render image for row", row)
        self.current_entry = self.entry_list[row][1]
        self.prepare_rendering()
        self.img_current_box.setTitle("Current: "+self.current_entry.value.filename)
        num_epochs = self.get_num_epochs(self.current_entry)
        self.epoch_slider.setMaximum(num_epochs)
        self.selected_epoch = min(self.selected_epoch, num_epochs-1)
        if self.selected_epoch == -1:
            self.selected_epoch = num_epochs-1
        self.epoch_slider.setValue(self.selected_epoch)
        self.epoch_slider_changed()
        self.slice_slider_changed()
        num_cameras = self.get_num_cameras(self.current_entry)
        self.camera_selection_spinbox.setRange(0, num_cameras-1)
        if self.camera_selection >= num_cameras:
            self.camera_selection = 0
            self.camera_selection_spinbox.setValue(0)

        # CHARTS

        rows = [i.row() for i in self.tableWidget.selectedIndexes()]
        rows = sorted(set(rows))
        print("Show plots for rows", rows)
        # remove old rows
        indices_to_remove = set(self.bar_series.keys()) - set(rows)
        for i in indices_to_remove:
            self.series.remove(self.bar_series[i])
            del self.bar_series[i]
        # add new
        indices_to_add = set(rows) - set(self.bar_series.keys())
        for i in indices_to_add:
            value = self.entry_list[row][1].value
            bset = QBarSet(value.filename)
            for l,lossname in enumerate(self.LossNames):
                loss = getattr(value, lossname)
                bset.append(loss[-1] if (loss is not None and len(loss)>0) else 0)
            self.bar_series[i] = bset
            self.series.append(bset)
        # adjust range
        max_loss = [
            (getattr(self.entry_list[r][1].value, lossname)[-1] if (getattr(self.entry_list[r][1].value, lossname) is not None and len(getattr(self.entry_list[r][1].value, lossname))>0) else 0)
                for lossname in self.LossNames
                for r in rows
        ]
        max_loss = np.max(max_loss)
        print("max loss:", max_loss)
        self.axisY.setRange(0, max_loss * 1.1)

        if len(rows)==1:
            # only one entry visible, switch to line chart
            for s in self.lineseries_list:
                self.linechart.removeSeries(s)
            self.lineseries_list.clear()
            value = self.entry_list[row][1].value
            max_loss = [
                np.max(getattr(value, lossname)) if getattr(value, lossname) is not None else 0
                for lossname in self.LossNames
            ]
            max_loss = np.max([l for l in max_loss if l>0])
            min_loss = [
                np.min(getattr(value, lossname)) if getattr(value, lossname) is not None else 100000000
                for lossname in self.LossNames
            ]
            min_loss = np.min([l for l in min_loss if l>0])
            self.lineAxisX.setRange(0, len(value[0]))
            self.lineAxisY.setRange(min_loss * 0.97, max_loss * 1.03)
            for idx,name in enumerate(self.LossNames):
                loss = getattr(value, name)
                if loss is None: continue
                if loss[0]==0: continue
                s = QLineSeries()
                for x,y in enumerate(loss):
                    s.append(x, y)
                #print(name, value[idx])
                s.setName(name)
                self.linechart.addSeries(s)
                s.attachAxis(self.lineAxisX)
                s.attachAxis(self.lineAxisY)
                self.lineseries_list.append(s)

            self.chart_stack.setCurrentIndex(1)
        else:
            self.chart_stack.setCurrentIndex(0)

    @abstractmethod
    def render_current_value(self, current_entry, current_epoch):
        ...

    def get_transfer_function(self, current_entry:Entry, current_epoch):
        raise NotImplementedError("Must be implemented by subclasses supporting transfer function selection")

    def epoch_slider_changed(self):
        if self.current_entry is None: return
        num_epochs = self.get_num_epochs(self.current_entry)
        self.selected_epoch = self.epoch_slider.value()
        self.selected_epoch = min(self.selected_epoch, num_epochs - 1)
        self.on_epoch_changed(self.current_entry, self.selected_epoch)
        self.epoch_label.setText("%d"%self.selected_epoch)
        img = self.render_current_value(self.current_entry, self.selected_epoch)
        self.img_current_pixmap = self.to_pixmap(img)
        if self.has_tf:
            tf = self.get_transfer_function(self.current_entry, self.selected_epoch)
            self.tf_current_pixmap = self.visualize_tf(tf, QPixmap(self.ImgRes, self.TFHeight))
        if self.has_volume_slices:
            self.slice_current_pixmap = self.to_pixmap(self.get_slice(
                False, self.current_entry, self.selected_epoch, self.current_slice, self.slice_axis))
        if self.vis_mode=='tf':
            self.current_label.setPixmap(self.tf_current_pixmap)
        elif self.vis_mode=='image':
            self.current_label.setPixmap(self.img_current_pixmap)
        elif self.vis_mode=='slices':
            self.current_label.setPixmap(self.slice_current_pixmap)

    def on_epoch_changed(self, current_entry, current_epoch):
        pass # overwritten in subclasses

    def get_slice(self, is_reference: bool, current_entry, current_epoch,
                  slice: float, axis: str):
        raise NotImplementedError("Must be implemented by subclasses supporting volume slices")

    def slice_slider_changed(self):
        if self.current_entry is None: return
        if not self.has_volume_slices: return
        self.current_slice = self.slice_slider.value() / 100.0
        self.slice_reference_pixmap = self.to_pixmap(self.get_slice(
            True, self.current_entry, 0, self.current_slice, self.slice_axis))
        self.slice_current_pixmap = self.to_pixmap(self.get_slice(
            False, self.current_entry, self.selected_epoch, self.current_slice, self.slice_axis))
        if self.vis_mode == 'slices':
            self.reference_label.setPixmap(self.slice_reference_pixmap)
            self.current_label.setPixmap(self.slice_current_pixmap)

    def slice_axis_changed(self, axis):
        self.slice_axis = axis
        self.slice_slider_changed() # redraw

    def switch_vis_mode(self, vis_mode):
        self.vis_mode = vis_mode
        if self.vis_mode=='tf':
            if self.tf_reference_pixmap is not None:
                self.reference_label.setPixmap(self.tf_reference_pixmap)
            if self.current_entry is not None:
                self.current_label.setPixmap(self.tf_current_pixmap)
        elif self.vis_mode=='image':
            if self.img_reference_pixmap is not None:
                self.reference_label.setPixmap(self.img_reference_pixmap)
            if self.current_entry is not None:
                self.current_label.setPixmap(self.img_current_pixmap)
        elif self.vis_mode == 'slices':
            self.reference_label.setPixmap(self.slice_reference_pixmap)
            self.current_label.setPixmap(self.slice_current_pixmap)
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if self.slice_slider is not None:
            self.slice_slider.setEnabled(self.vis_mode == 'slices')

    def switch_background_mode(self, mode):
        if mode=="white":
            self.white_background = True
        else:
            self.white_background = False
        self.epoch_slider_changed()
        self.visualize_reference()

    def send_image_to_clipboard(self, pixmap:QPixmap):
        import win32clipboard
        from io import BytesIO
        output = BytesIO()
        raise NotImplementedError("TODO: convert pixmap to BytesIO")

        data = output.getvalue()[14:]
        output.close()
        win32clipboard.OpenClipboard()
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardData(win32clipboard.CF_DIB, data)
        win32clipboard.CloseClipboard()

    def save_reference(self):
        print("Save reference")
        if self.vis_mode=='tf':
            pixmap = self.tf_reference_pixmap
            preferredFilename = "reference_tf.png"
        elif self.vis_mode=='image':
            pixmap = self.img_reference_pixmap
            preferredFilename = "reference_img.png"
        elif self.vis_mode=='slices':
            pixmap = self.slice_reference_pixmap
            preferredFilename = "reference_slice%02d.png"%int(self.current_slice*100)
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if pixmap is None:
            print("pixmap is none")
            return
        filename = QFileDialog.getSaveFileName(
            self.window, "Save reference rendering",
            os.path.join(self.save_folder, preferredFilename),
            self.ExportFileNames)[0]
        if filename is not None and len(filename)>0:
            pixmap.save(filename)
            print("Saved to", filename)

    def save_current(self):
        if self.current_entry is None: return
        print("Save current")
        if self.vis_mode=='tf':
            pixmap = self.tf_current_pixmap
            preferredFilename = self.current_entry.value.filename + "_epoch%03d_tf.png"%self.selected_epoch
        elif self.vis_mode=='image':
            pixmap = self.img_current_pixmap
            preferredFilename = self.current_entry.value.filename + "_epoch%03d_img.png" % self.selected_epoch
        elif self.vis_mode=='slices':
            pixmap = self.slice_current_pixmap
            preferredFilename = self.current_entry.value.filename + "_epoch%03d_slice%02d.png" % (
                self.selected_epoch, int(self.current_slice*100))
        else:
            raise ValueError("Unknown vis mode: " + self.vis_mode)
        if pixmap is None:
            print("pixmap is none")
            return
        filename = QFileDialog.getSaveFileName(
            self.window, "Save current rendering",
            os.path.join(self.save_folder, preferredFilename),
            self.ExportFileNames)[0]
        if filename is not None and len(filename) > 0:
            pixmap.save(filename)
            print("Saved to", filename)

    def camera_follows_settings_changed(self):
        if self.camera_follows_settings_button.isChecked():
            self.camera_follows_settings = True
            pitchYawDistance = self.get_camera_settings(
                self.current_entry, self.camera_selection)
            self._updateCamera(pitchYawDistance, True, True)
        else:
            self.camera_follows_settings = False
    def camera_yaw_or_pitch_changed(self):
        if self.camera_disable_slider_events: return
        self.camera_follows_settings = False
        self.camera_follows_settings_button.setChecked(False)
        yaw = np.deg2rad(self.camera_yaw_slider.value())
        pitch = np.deg2rad(self.camera_pitch_slider.value())
        distance = self.current_entry.pitch_yaw_distances[0].z
        pitchYawDistance = pyrenderer.double3(pitch, yaw, distance)
        self._updateCamera(pitchYawDistance, True, False)
    def camera_selection_changed(self):
        self.camera_selection = self.camera_selection_spinbox.value()
        self.camera_selection = min(self.camera_selection, self.get_num_cameras(self.current_entry)-1)
        self.camera_follows_settings_button.setChecked(True)
        self.camera_follows_settings = True
        pitchYawDistance = self.get_camera_settings(
            self.current_entry, self.camera_selection)
        self._updateCamera(pitchYawDistance, True, True)

    def _export_table(self):
        ExportFileNames = "TSV (*.tsv)"
        filename = QFileDialog.getSaveFileName(
            self.window, "Export table as .tsv",
            os.path.join(self.save_folder, "results.tsv"),
            ExportFileNames)[0]
        if filename is not None and len(filename) > 0:
            self.exportAsFile(filename, sep='\t', newline='\n')
            print("Saved to", filename)

    def create_browser(self, parent):
        parentLayout = QVBoxLayout(parent)
        buttonRowLayout = QHBoxLayout(parent)
        self.reload_button = QPushButton("Reload", parent)
        self.reload_button.clicked.connect(lambda: self.reparse())
        buttonRowLayout.addWidget(self.reload_button)
        self.export_button = QPushButton("Export", parent)
        self.export_button.clicked.connect(lambda: self._export_table())
        buttonRowLayout.addWidget(self.export_button)
        parentLayout.addLayout(buttonRowLayout)
        self.tableWidget = QTableWidget(parent)
        self.tableWidget.setColumnCount(len(self.KeyNames)+len(self.LossNames))
        # header
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.tableWidget.setHorizontalHeaderLabels(self.KeyNames+self.LossNames)
        # layout
        parentLayout.addWidget(self.tableWidget, stretch=1)
        parent.setLayout(parentLayout)
        # event
        self.tableWidget.cellClicked.connect(self.selection_changed)


    def create_charts(self, parent):
        parentLayout = QHBoxLayout(parent)
        self.chart_stack = QStackedWidget(parent)

        self.chart = QChart()
        self.series = QBarSeries()
        self.chart.addSeries(self.series)
        self.axisX = QBarCategoryAxis()
        self.axisX.append(self.LossNames)
        self.chart.addAxis(self.axisX, Qt.AlignBottom)
        self.series.attachAxis(self.axisX)
        self.axisY = QValueAxis()
        self.chart.addAxis(self.axisY, Qt.AlignLeft)
        self.series.attachAxis(self.axisY)
        self.chart.legend().setVisible(True)
        self.chart.legend().setAlignment(Qt.AlignRight)

        self.chartView = QChartView(self.chart, parent)
        self.chartView.setRenderHint(QPainter.Antialiasing)
        self.chart_stack.addWidget(self.chartView)

        self.linechart = QChart()
        self.lineAxisX = QValueAxis()
        self.lineAxisY = QLogValueAxis()
        #self.lineAxisY = QValueAxis()
        self.lineAxisX.setMinorGridLineVisible(True)
        self.lineAxisX.setMinorTickCount(10)
        self.linechart.addAxis(self.lineAxisX, Qt.AlignBottom)
        self.linechart.addAxis(self.lineAxisY, Qt.AlignLeft)
        self.linechart.legend().setVisible(True)
        self.linechart.legend().setAlignment(Qt.AlignRight)
        self.lineChartView = QChartView(self.linechart, parent)
        self.lineChartView.setRenderHint(QPainter.Antialiasing)
        self.chart_stack.addWidget(self.lineChartView)

        parentLayout.addWidget(self.chart_stack)

    def _custom_image_controls(self, parentLayout, parentWidget):
        pass

    def _image_control_hook(self, layout, parent, type: str):
        """
        Hook into the HBoxLayout-creation for
         - image/epoch (type='epoch')
         - slice (type='slice')
         - camera (type='camera')
        """
        pass

    def create_images(self, parent):
        parentLayout = QVBoxLayout(parent)

        layout1 = QHBoxLayout(parent)

        self.vis_mode_button_group = QButtonGroup(parent)
        self.radio_img = QRadioButton("Image", parent)
        if self.vis_mode=='image':
            self.radio_img.setChecked(True)
        layout1.addWidget(self.radio_img)
        self.vis_mode_button_group.addButton(self.radio_img)
        self.radio_img.clicked.connect(lambda: self.switch_vis_mode('image'))
        if self.has_tf:
            self.radio_tf = QRadioButton("TF", parent)
            if self.vis_mode=='tf':
                self.radio_tf.setChecked(True)
            layout1.addWidget(self.radio_tf)
            self.vis_mode_button_group.addButton(self.radio_tf)
            self.radio_tf.clicked.connect(lambda: self.switch_vis_mode('tf'))
        if self.has_volume_slices:
            self.radio_slices = QRadioButton("Slices", parent)
            if self.vis_mode == 'slices':
                self.radio_slices.setChecked(True)
            layout1.addWidget(self.radio_slices)
            self.vis_mode_button_group.addButton(self.radio_slices)
            self.radio_slices.clicked.connect(lambda: self.switch_vis_mode('slices'))

        self.background_button_group = QButtonGroup(parent)
        layout1.addWidget(QLabel("Background:"))
        self.background_white_button = QRadioButton("White", parent)
        if self.white_background:
            self.background_white_button.setChecked(True)
        layout1.addWidget(self.background_white_button)
        self.background_button_group.addButton(self.background_white_button)
        self.background_black_button = QRadioButton("Black", parent)
        if not self.white_background:
            self.background_black_button.setChecked(True)
        layout1.addWidget(self.background_black_button)
        self.background_button_group.addButton(self.background_black_button)
        self.background_white_button.clicked.connect(lambda: self.switch_background_mode('white'))
        self.background_black_button.clicked.connect(lambda: self.switch_background_mode('black'))

        layout1.addWidget(QLabel("Epoch:"))
        self.epoch_slider = QSlider(Qt.Horizontal, parent)
        self.epoch_slider.setMinimum(0)
        self.epoch_slider.setTracking(True)
        self.epoch_slider.valueChanged.connect(self.epoch_slider_changed)
        layout1.addWidget(self.epoch_slider)
        self.epoch_label = QLabel("0", parent)
        layout1.addWidget(self.epoch_label)

        self._image_control_hook(layout1, parent, 'epoch')
        parentLayout.addLayout(layout1)

        if self.allows_free_camera:
            layout4 = QHBoxLayout(parent)
            layout4.addWidget(QLabel("Camera:"))

            self.camera_follows_settings_button = QCheckBox("Follows settings", parent)
            self.camera_follows_settings_button.setChecked(self.camera_follows_settings)
            self.camera_follows_settings_button.clicked.connect(
                lambda: self.camera_follows_settings_changed())
            layout4.addWidget(self.camera_follows_settings_button)

            self.camera_selection_spinbox = QSpinBox(parent)
            self.camera_selection_spinbox.setRange(0, 100)
            self.camera_selection_spinbox.valueChanged.connect(
                lambda: self.camera_selection_changed())
            layout4.addWidget(self.camera_selection_spinbox)

            layout4.addWidget(QLabel("Yaw:"))
            self.camera_yaw_slider = QSlider(Qt.Horizontal, parent)
            self.camera_yaw_slider.setMinimum(0)
            self.camera_yaw_slider.setMaximum(360)
            self.camera_yaw_slider.setTracking(True)
            self.camera_yaw_slider.valueChanged.connect(self.camera_yaw_or_pitch_changed)
            layout4.addWidget(self.camera_yaw_slider)

            layout4.addWidget(QLabel("Pitch:"))
            self.camera_pitch_slider = QSlider(Qt.Horizontal, parent)
            self.camera_pitch_slider.setMinimum(-80)
            self.camera_pitch_slider.setMaximum(80)
            self.camera_pitch_slider.setTracking(True)
            self.camera_pitch_slider.valueChanged.connect(self.camera_yaw_or_pitch_changed)
            layout4.addWidget(self.camera_pitch_slider)

            self._image_control_hook(layout4, parent, 'camera')
            parentLayout.addLayout(layout4)

        if self.has_volume_slices:
            layout3 = QHBoxLayout(parent)

            layout3.addWidget(QLabel("Axis:"))
            self.slice_axis_button_group = QButtonGroup(parent)

            self.radio_axis_x = QRadioButton("X", parent)
            if self.slice_axis == 'x':
                self.radio_axis_x.setChecked(True)
            layout3.addWidget(self.radio_axis_x)
            self.slice_axis_button_group.addButton(self.radio_axis_x)
            self.radio_axis_x.clicked.connect(lambda: self.slice_axis_changed('x'))

            self.radio_axis_y = QRadioButton("Y", parent)
            if self.slice_axis == 'y':
                self.radio_axis_y.setChecked(True)
            layout3.addWidget(self.radio_axis_y)
            self.slice_axis_button_group.addButton(self.radio_axis_y)
            self.radio_axis_y.clicked.connect(lambda: self.slice_axis_changed('y'))

            self.radio_axis_z = QRadioButton("Z", parent)
            if self.slice_axis == 'z':
                self.radio_axis_z.setChecked(True)
            layout3.addWidget(self.radio_axis_z)
            self.slice_axis_button_group.addButton(self.radio_axis_z)
            self.radio_axis_z.clicked.connect(lambda: self.slice_axis_changed('z'))

            layout3.addWidget(QLabel("Slice:"))
            self.slice_slider = QSlider(Qt.Horizontal, parent)
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(100)
            self.slice_slider.setTracking(True)
            self.slice_slider.valueChanged.connect(self.slice_slider_changed)
            layout3.addWidget(self.slice_slider)
            parentLayout.addLayout(layout3)
            if not self.vis_mode == 'slices':
                self.slice_slider.setEnabled(False)

            self._image_control_hook(layout3, parent, 'slice')
            parentLayout.addLayout(layout3)
        else:
            self.slice_slider = None

        self._custom_image_controls(parentLayout, parent)

        layout2 = QHBoxLayout(parent)
        layout2.addStretch(1)

        box1 = QGroupBox(parent)
        box1.setTitle("Reference")
        box1Layout = QHBoxLayout(box1)
        self.reference_label = QLabel(box1)
        self.reference_label.setFixedSize(self.ImgRes, self.ImgRes)
        self.reference_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.reference_label.customContextMenuRequested.connect(lambda e: self.save_reference())
        box1Layout.addWidget(self.reference_label)
        layout2.addWidget(box1)

        layout2.addStretch(1)

        box2 = QGroupBox(parent)
        box2.setTitle("Current")
        self.img_current_box = box2
        box2Layout = QHBoxLayout(box2)
        self.current_label = QLabel(box1)
        self.current_label.setFixedSize(self.ImgRes, self.ImgRes)
        self.current_label.setContextMenuPolicy(Qt.CustomContextMenu)
        self.current_label.customContextMenuRequested.connect(lambda e: self.save_current())
        box2Layout.addWidget(self.current_label)
        layout2.addWidget(box2)

        layout2.addStretch(1)
        parentLayout.addLayout(layout2, 1)
        parent.setLayout(parentLayout)


    def vis(self):
        self.a = QApplication([])
        self.window = QMainWindow()

        sizePolicyXY = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sizePolicyXY.setHorizontalStretch(0)
        sizePolicyXY.setVerticalStretch(0)
        sizePolicyX = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        sizePolicyX.setHorizontalStretch(0)
        sizePolicyX.setVerticalStretch(0)
        sizePolicyY = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        sizePolicyY.setHorizontalStretch(0)
        sizePolicyY.setVerticalStretch(0)

        self.window.setSizePolicy(sizePolicyXY)

        centralWidget = QWidget(self.window)
        gridLayout = QGridLayout(centralWidget)

        splitter2 = QSplitter(centralWidget)
        splitter2.setOrientation(Qt.Horizontal)
        splitter2.setChildrenCollapsible(False)
        splitter2.setSizePolicy(sizePolicyXY)

        browserBox = QGroupBox(splitter2)
        browserBox.setTitle("Parameters")
        browserBox.setSizePolicy(sizePolicyY)
        self.create_browser(browserBox)
        splitter2.addWidget(browserBox)

        splitter1 = QSplitter(splitter2)
        splitter1.setOrientation(Qt.Vertical)
        splitter1.setChildrenCollapsible(False)
        splitter1.setSizePolicy(sizePolicyXY)

        chartsBox = QGroupBox(splitter1)
        chartsBox.setTitle("Charts")
        chartsBox.setSizePolicy(sizePolicyY)
        self.create_charts(chartsBox)
        splitter1.addWidget(chartsBox)

        visBox = QGroupBox(splitter1)
        visBox.setTitle("Images")
        visBox.setSizePolicy(sizePolicyY)
        self.create_images(visBox)
        splitter1.addWidget(visBox)
        splitter2.addWidget(splitter1)
        splitter1.setStretchFactor(0, 1)
        splitter2.setStretchFactor(0, 1)

        gridLayout.addWidget(splitter2)
        self.window.setCentralWidget(centralWidget)
        self.window.resize(1024, 768)
        print("UI created")


if __name__ == "__main__":
    ui = UI(os.path.join(os.getcwd(), "..\\..\\results\\tf\\meta"))
    ui.show()
