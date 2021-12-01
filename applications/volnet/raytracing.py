"""
PyTorch-emulation of ray tracing
"""

import numpy as np
import torch
from typing import Union, List, Any, Tuple, Optional
import tqdm

import common.utils as utils
import pyrenderer

class Raytracing:

    def __init__(self,
                 image_evaluator: pyrenderer.IImageEvaluator,
                 network_output : str,
                 stepsize: float,
                 image_width: int, image_height: int,
                 dtype, device):
        """
        Creates a new raytracer instance using the settings from the image_evaluator.

        It assumes, extracts and exposes the following modules from the image evaluator:
         - camera (for the ray generation)
         - ray_evaluator.volume (to obtain the bounding box)
         - ray_evaluator.tf (for shading, only for network_output=='density')

        :param settings: the renderer settings (screen size, box size)
        :param fov_y_radians: the fov for the camera
        :param network_output: the network output mode, either 'color' or 'density'
        :param stepsize: the stepsize in world space
        """
        assert network_output in ["rgbo", "density"]
        self._image_evaluator = image_evaluator
        self._network_output = network_output
        self._stepsize = stepsize
        self._image_width = image_width
        self._image_height = image_height

        self._dtype = dtype
        self._device = device

        self._camera = image_evaluator.camera
        self._volume_interpolation = image_evaluator.volume
        if network_output=='density':
            self._tf = image_evaluator.ray_evaluator.tf
            self._min_density = image_evaluator.ray_evaluator.min_density
            self._max_density = image_evaluator.ray_evaluator.max_density
        else:
            self._tf = None

        box_min = self._volume_interpolation.box_min()
        box_size = self._volume_interpolation.box_size()
        box_min = torch.from_numpy(utils.cvector_to_numpy(box_min))
        box_size = torch.from_numpy(utils.cvector_to_numpy(box_size))
        self._box_min = box_min.unsqueeze(0).to(device=device, dtype=dtype)
        self._box_size = box_size.unsqueeze(0).to(device=device, dtype=dtype)

    def stepsize(self) -> float:
        return self._stepsize
    def set_stepsize(self, s):
        self._stepsize = s
    def set_resolution(self, width, height):
        self._image_width = width
        self._image_height = height
    def camera(self) -> pyrenderer.ICamera:
        return self._camera
    def volume(self) -> pyrenderer.IVolumeInterpolation:
        return self._volume_interpolation
    def tf(self) -> pyrenderer.ITransferFunction:
        return self._tf
    def box_min(self) -> torch.Tensor:
        return self._box_min
    def box_size(self) -> torch.Tensor:
        return self._box_size

    @staticmethod
    def intersection_aabb(
            ray_start, ray_dir, box_min, box_size):
        """
        Computes AABB-intersection. All tensors are of shape (B,3)
        with broadcasting over the batches
        :return: (tmin, tmax) of shape (B,1)
        """

        inv_ray_dir = 1.0 / ray_dir
        t135 = (box_min - ray_start) * inv_ray_dir
        t246 = (box_min + box_size - ray_start) * inv_ray_dir
        tmin = torch.max(torch.minimum(t135, t246), dim=1, keepdim=True)[0]
        tmax = torch.min(torch.maximum(t135, t246), dim=1, keepdim=True)[0]
        return tmin, tmax

    @staticmethod
    def generate_camera_ray(camera: pyrenderer.ICamera, width: int, height:int, dtype, *, multisampling:int=0):
        if dtype==torch.float32:
            double_precision = False
        elif dtype==torch.float64:
            double_precision = True
        else:
            raise ValueError(f"Unknown dtype {dtype}")
        if multisampling==0:
            camera_ray_start, camera_ray_dir = camera.generate_rays(
                width, height, double_precision)
        else:
            camera_ray_start, camera_ray_dir = camera.generate_rays_multisampling(
                width, height, multisampling, double_precision)
        B, H, W, _ = camera_ray_start.shape
        # reshape to (B*H*W)*3
        #camera_ray_start = camera_ray_start.view(B * H * W, 3)
        #camera_ray_dir = camera_ray_dir.view(B * H * W, 3)
        return B, H, W, camera_ray_start, camera_ray_dir


    def _find_entry_exit(self, camera_ray_start, camera_ray_dir):
        tmin, tmax = Raytracing.intersection_aabb(
            camera_ray_start, camera_ray_dir, self._box_min, self._box_size)
        return self._box_min, self._box_size, tmin, tmax


    def _predict(self, volume_pos: torch.Tensor,
                  camera_ray_dir: torch.Tensor,
                  network: torch.nn.Module,
                  tf: pyrenderer.ITransferFunction,
                  network_args: List[Any],
                  previous_prediction: Optional[torch.Tensor] = None,
                  stepsize: Optional[float] = None):

        if network.use_direction():
            volume_pos = torch.cat((volume_pos, camera_ray_dir), dim=1)
        prediction = network(volume_pos, *network_args)
        ##DEBUG
        #middle = torch.tensor([[0.5,0.5,0.5]], dtype=volume_pos.dtype, device=volume_pos.device)
        #color = 0.5 * torch.ones_like(volume_pos)
        #absorption = torch.clamp(0.5 - torch.linalg.norm((middle-volume_pos), dim=1, keepdim=True), min=0)
        #prediction = torch.cat((color, absorption), dim=1)

        BHW = prediction.shape[0]
        if self._network_output == "rgbo":
            assert prediction.shape == (BHW, 4), "wrong output shape %s" % str(prediction.shape)
            color = prediction
            if stepsize is not None:
                color = torch.cat([color[:,:3], color[:,3:]*stepsize], dim=1)
        elif self._network_output == "density":
            assert prediction.shape == (BHW, 1), "wrong output shape %s" % str(prediction.shape)
            assert tf is not None, \
                "if the network predictions densities, the TF must be specified"
            prediction = prediction.to(dtype=self._dtype)
            if stepsize is None:
                color = tf.evaluate(prediction, self._min_density, self._max_density)
            else:
                if previous_prediction is None:
                    previous_prediction = torch.full_like(prediction, -1)
                color = tf.evaluate_with_previous(prediction, self._min_density, self._max_density, previous_prediction, stepsize)
        else:
            raise ValueError("Unknown network prediction mode")
        return color, prediction.detach()

    def _blend(self, prev_color, prev_alpha, current_color, mask, stepsize: Optional[float]):
        if stepsize is None: stepsize = self._stepsize
        current_alpha = 1 - torch.exp(-current_color[:, 3:] * stepsize)
        zeros = torch.zeros(1,1, dtype=current_alpha.dtype, device=current_alpha.device)
        current_alpha = torch.where(mask, current_alpha, zeros)  # out of bounds
        next_color = prev_color + (1 - prev_alpha) * current_color[:, :3] * current_alpha
        next_alpha = prev_alpha + (1 - prev_alpha) * current_alpha
        return next_color, next_alpha

    def _inverse_blend(self,
                      next_color, next_alpha, current_color, mask,
                      grad_next_color, grad_next_alpha,
                      stepsize: Optional[float]):

        current_rgb = current_color[:, :3]
        if stepsize is None: stepsize = self._stepsize
        current_alpha = 1 - torch.exp(-current_color[:, 3:] * stepsize)
        zeros = torch.zeros(1, 1, dtype=current_alpha.dtype, device=current_alpha.device)
        current_alpha = torch.where(mask, current_alpha, zeros)  # out of bounds

        # this would be the contents of the forward pass:
        #next_color = prev_color + (1 - prev_alpha) * current_rgb * current_alpha
        #next_alpha = prev_alpha + (1 - prev_alpha) * current_alpha

        # invert to reconstruct the accumulator before blending
        prev_alpha = (current_alpha - next_alpha) / (current_alpha - 1)
        prev_color = next_color - (1 - prev_alpha) * current_rgb * current_alpha

        # ADJOINT

        #>> next_alpha = prev_alpha + (1 - prev_alpha) * current_alpha;
        grad_current_alpha = grad_next_alpha * (1 - prev_alpha)
        grad_prev_alpha = grad_next_alpha * (1 - current_alpha)

        #>> next_color = prev_color + (1 - prev_alpha) * current_rgb * current_alpha
        def dot(t1, t2):
            return torch.sum(t1*t2, dim=1, keepdim=True)
        grad_current_alpha += dot(grad_next_color, current_rgb - current_rgb * prev_alpha)
        grad_prev_alpha += dot(grad_next_color, -current_rgb * current_alpha)
        grad_current_rgb = grad_next_color * (current_alpha * (1 - prev_alpha))
        grad_prev_color = grad_next_color

        #>> current_alpha = torch.where(mask, current_alpha, zeros)
        grad_current_alpha.masked_fill_(mask.bitwise_not(), 0.0)
        #>> current_alpha = 1 - torch.exp(-current_color[:, 3:] * self.stepsize)
        grad_current_alpha = grad_current_alpha * stepsize * torch.exp(-current_color[:, 3:] * self._stepsize)

        grad_current_color = torch.cat((grad_current_rgb, grad_current_alpha), dim=1)
        return prev_color, prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color

    def _generate_camera_rays(self, camera, multisampling:int=0):
        if isinstance(camera, torch.Tensor):
            # camera specified using parameters -> set parameters
            cam2 = self.camera()
            cam2.set_parameters(camera)
            camera = cam2
        if isinstance(camera, (tuple, list)):
            assert multisampling==0
            camera_ray_start, camera_ray_dir = camera
            B, H, W, _ = camera_ray_start.shape
        else:
            B, H, W, camera_ray_start, camera_ray_dir = \
                Raytracing.generate_camera_ray(camera, self._image_width, self._image_height, self._dtype, multisampling=multisampling)
        return B, H, W, camera_ray_start, camera_ray_dir

    def get_max_steps(self, camera: Union[pyrenderer.ICamera, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None):
        """
        Computes the maximal number of steps through the volume
        from the given camera with the current stepsize (see self.set_stepsize)
        :param camera: the camera to use. If None, self.camera() is used.
        :return: the maximal number of steps
        """
        if camera is None:
            camera = self.camera()

        B, H, W, camera_ray_start, camera_ray_dir = \
            self._generate_camera_rays(camera)
        # reshape to (B*H*W)*3
        camera_ray_start = camera_ray_start.view(B * H * W, 3)
        camera_ray_dir = camera_ray_dir.view(B * H * W, 3)

        # find entry and exit times, shape = (3,)
        box_min, box_size, tmin, tmax = \
            self._find_entry_exit(camera_ray_start, camera_ray_dir)

        max_steps = int(torch.max(tmax - tmin).item() / self._stepsize)

        return max_steps

    def full_trace_forward(self,
                           network: torch.nn.Module,
                           camera: Union[pyrenderer.ICamera, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
                           tf: pyrenderer.ITransferFunction = None,
                           is_half = False,
                           network_args: List[Any] = None):
        """
        Full differentiable ray tracing without checkpointing.
        :param network: the scene representation network
        :param camera: the camera to use. If None, self.camera() is used
        :param tf: the TF to use. If None, self.tf() is used
        :param is_half: half-precision network evaluation?
        :return: the output image (B,C,H,W)
        """

        if camera is None:
            camera = self.camera()
        if tf is None:
            tf = self.tf()

        B, H, W, camera_ray_start, camera_ray_dir = \
            self._generate_camera_rays(camera)
        toCHW = True

        return self._full_trace_forward(network, camera_ray_start, camera_ray_dir,
                                        tf, is_half, toCHW, network_args)

    def _full_trace_forward(self,
                           network: torch.nn.Module,
                           camera_ray_start: torch.Tensor,
                           camera_ray_dir: torch.Tensor,
                           tf: pyrenderer.ITransferFunction,
                           is_half: bool,
                           toCHW: bool,
                           network_args: List[Any]):

        B, H, W, _ = camera_ray_start.shape
        # reshape to (B*H*W)*3
        camera_ray_start = camera_ray_start.view(B * H * W, 3)
        camera_ray_dir = camera_ray_dir.view(B * H * W, 3)

        def expand_arg(a):
            try:
                a = a.expand((-1,H*W)).reshape(B*H*W)
            except AttributeError:
                pass
            return a
        network_args = [expand_arg(a) for a in network_args]

        # find entry and exit times, shape = (3,)
        box_min, box_size, tmin, tmax = \
            self._find_entry_exit(camera_ray_start, camera_ray_dir)

        max_steps = int(torch.max(tmax-tmin).item() / self._stepsize)
        #print("Raytrace with", max_steps, "steps")

        # perform stepping
        final_color = torch.zeros((B * H * W, 3), dtype=self._dtype, device=self._device)
        final_alpha = torch.zeros((B * H * W, 1), dtype=self._dtype, device=self._device)
        previous_prediction = None
        for t in range(max_steps):
            tcurrent = tmin + t*self._stepsize
            world_pos = camera_ray_start + tcurrent * camera_ray_dir
            volume_pos = (world_pos - box_min) / box_size
            # call network
            if is_half:
                color, prediction = self._predict(volume_pos.half(), camera_ray_dir.half(), network, tf, network_args, previous_prediction, self._stepsize)
                color = color.float()
                prediction = prediction.float()
            else:
                color, prediction = self._predict(volume_pos, camera_ray_dir, network, tf, network_args, previous_prediction, self._stepsize)
            previous_prediction = prediction # for pre-integration
            # blending
            final_color, final_alpha = self._blend(
                final_color, final_alpha, color, tcurrent < tmax, stepsize=1) # stepspize scaling already done in _predict

        # done, reshape back to original
        rgba = torch.cat((final_color, final_alpha), dim=1)
        if toCHW:
            return utils.toCHW(rgba.view(B, H, W, 4))
        else:
            return rgba.view(B, H, W, 4)

    def monte_carlo_trace(self,
                          network: torch.nn.Module,
                          num_samples: int,
                          camera: Union[pyrenderer.ICamera, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], None] = None,
                          tf: pyrenderer.ITransferFunction = None,
                          is_half=False,
                          network_args: List[Any] = None):
        if camera is None:
            camera = self.camera()
        if tf is None:
            tf = self.tf()
        ray_evaluator = self._image_evaluator.ray_evaluator
        assert isinstance(ray_evaluator, pyrenderer.RayEvaluationMonteCarlo)

        # read parameters from the ray evaluator and tf
        num_bounces = ray_evaluator.num_bounces
        max_absorption = tf.get_max_absorption()
        div_max_absorption = 1 / max_absorption
        density_min = self._min_density
        density_max = self._max_density
        div_density_range = 1 / (density_max - density_min)
        color_scaling = ray_evaluator.color_scaling
        light_intensity = ray_evaluator.light_intensity

        B, H, W, camera_ray_start, camera_ray_dir = \
            self._generate_camera_rays(camera, multisampling=num_samples)
        time = 1
        toCHW = True
        device = camera_ray_start.device
        dtype = camera_ray_start.dtype

        def expand_arg(a):
            try:
                a = a.expand((B,H*W)).reshape(B*H*W)
            except AttributeError:
                pass
            return a
        network_args = [expand_arg(a) for a in network_args]

        def delta_tracking(start, dir, valid_mask):
            is_valid = valid_mask
            tcurrent = torch.zeros((BHW,1), device=device, dtype=dtype)
            previous_prediction = None
            out_position = torch.zeros((BHW,3), device=device, dtype=dtype)
            out_color = torch.zeros((BHW,4), device=device, dtype=dtype)
            has_hit = torch.zeros((BHW,1), device=device, dtype=torch.bool)
            MAX_ITERATIONS = 500
            for _ in range(MAX_ITERATIONS):
                # sample in homogeneous medium
                neg_stepsize = torch.log(torch.rand_like(tcurrent)) * div_max_absorption
                tcurrent = tcurrent - neg_stepsize
                world_pos = start + dir * tcurrent
                # bbox transformation and update position
                volume_pos = (world_pos - box_min) / box_size
                is_inside = (volume_pos>=0.0) & (volume_pos<=1.0)
                is_inside = is_inside[:,0:1] & is_inside[:,1:2] & is_inside[:,2:3]
                out_position = torch.where(is_valid, world_pos, out_position)
                is_valid = is_valid & is_inside
                # evaluate
                if is_half:
                    color, prediction = self._predict(volume_pos.half(), dir.half(), network, tf,
                                                      network_args, previous_prediction, stepsize=1)
                    color = color.float()
                    prediction = prediction.float()
                else:
                    color, prediction = self._predict(volume_pos, dir, network, tf, network_args,
                                                      previous_prediction, stepsize=1)
                previous_prediction = prediction  # for pre-integration
                # check if this was a virtual particle or real particle
                p = torch.rand_like(tcurrent)
                current_absorption = color[:,3:]
                is_real = current_absorption*div_max_absorption>p
                has_hit = has_hit | (is_valid & is_real)
                out_color = torch.where(is_valid&is_real, color, out_color)
                # termination
                is_valid = is_valid & torch.logical_not(is_real)
                #print(f"valid: {torch.mean(is_valid*1.0).item()}, has_hit: {torch.mean(has_hit*1.0).item()}")
                if not torch.any(is_valid): break
            return out_position, out_color, has_hit

        def normalize(a: torch.Tensor, axis=-1):
            l2 = torch.norm(a, dim=axis, keepdim=True)
            l2 = torch.clip(l2, min=1e-5)
            return a / l2

        total_out = torch.zeros((H*W,4), dtype=dtype, device=device)
        for sample in tqdm.trange(num_samples):

            # reshape to (B*H*W)*3
            local_camera_ray_start = camera_ray_start[sample].view(H * W, 3)
            local_camera_ray_dir = camera_ray_dir[sample].view(H * W, 3)
            BHW = H*W

            # find entry and exit times, shape = (3,)
            box_min, box_size, tmin, tmax = \
                self._find_entry_exit(local_camera_ray_start, local_camera_ray_dir)

            emission = torch.zeros((BHW, 3), device=device, dtype=dtype)
            beta = torch.ones((BHW, 3), device=device, dtype=dtype)
            out_alpha = torch.empty((BHW, 3), device=device, dtype=dtype)
            position = local_camera_ray_start + (tmin+0.0001) * local_camera_ray_dir
            direction = local_camera_ray_dir

            zero = torch.zeros((BHW, 1), device=device, dtype=dtype)
            one = torch.ones((BHW, 1), device=device, dtype=dtype)

            is_valid = tmin < tmax
            for bounce in range(num_bounces+1):
                # find next event
                next_position, out_color, has_hit = delta_tracking(position, direction, is_valid)
                out_rgb = out_color[:,:3]
                out_absorption = out_color[:,3:]

                # has first intersection? Determines alpha value
                if bounce==0:
                    out_alpha = torch.where(has_hit, one, zero)

                # medium intersection
                beta = torch.where(has_hit, beta * out_rgb * (out_absorption * color_scaling), beta)

                # direct illumination
                light_pos = pyrenderer.RayEvaluationMonteCarlo.SampleLight(self._image_evaluator, BHW, time); time+=1
                light_pos = light_pos.view(BHW, 3)
                light_dir = normalize(light_pos - next_position)
                p = pyrenderer.RayEvaluationMonteCarlo.PhaseFunctionProbability(self._image_evaluator, direction, light_dir, next_position)
                lightdt_position, lightdt_color, lightdt_hit = delta_tracking(next_position, light_dir, has_hit)
                has_light = has_hit & torch.logical_not(lightdt_hit)
                #print(f"max p: {torch.max(p).item()}, light intensity: {light_intensity}")
                emission = torch.where(has_light, emission + beta * (p*light_intensity), emission)

                # next ray
                next_dir, beta_scaling = pyrenderer.RayEvaluationMonteCarlo.NextDirection(self._image_evaluator, next_position, direction, time); time+=1
                beta = beta * beta_scaling
                direction = next_dir
                position = next_position

                is_valid = has_hit
                #if not torch.any(is_valid): break # early out

            # done
            total_out = total_out + torch.cat((emission, out_alpha), dim=1)
        total_out = total_out.view(1, H, W, 4)
        # average samples
        total_out = total_out / num_samples

        if toCHW:
            return utils.toCHW(total_out)
        return total_out


    def checkpointed_trace(self,
                           network: torch.nn.Module,
                           camera: Union[pyrenderer.ICamera, torch.Tensor, None] = None,
                           tf: pyrenderer.ITransferFunction = None,
                           is_half=False,
                           network_args: List[Any] = None):
        """
        Same syntax and semantic as self.full_trace_forward,
        but computes the gradient step by step via local inversion.
        This is a bit more costly in terms of performance, but does not require additional memory
        :param network: the scene representation network
        :param camera: the camera to use. If None, self.camera() is used
        :param tf: the TF to use. If None, self.tf() is used
        :param is_half: half-precision network evaluation?
        :return: the output image (B,C,H,W)
        """

        if camera is None:
            camera = self.camera()
        if tf is None:
            tf = self.tf()

        # generate camera rays
        if isinstance(camera, torch.Tensor):
            # camera specified using parameters -> set parameters
            cam2 = self.camera()
            cam2.set_parameters(camera)
            camera = cam2
        if isinstance(camera, (tuple, list)):
            camera_ray_start, camera_ray_dir = camera
            B, H, W, _ = camera_ray_start.shape
        else:
            B, H, W, camera_ray_start, camera_ray_dir = \
                Raytracing.generate_camera_ray(camera, self._image_width, self._image_height, self._dtype)
        toCHW = True

        # ugly hack, but without an input with requires_grad=True,
        # the autograd.Function is not recorded
        camera_ray_start0 = camera_ray_start.requires_grad_()
        camera_ray_dir0 = camera_ray_dir.requires_grad_()

        out = _CheckpointedTrace.apply(self, network, camera_ray_start0, camera_ray_dir0,
                                       tf, is_half, toCHW, network_args)
        return out

class _CheckpointedTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self: Raytracing,
                network: torch.nn.Module,
                camera_ray_start: torch.Tensor,
                camera_ray_dir: torch.Tensor,
                tf: pyrenderer.ITransferFunction,
                is_half: bool,
                toCHW: bool,
                network_args):

        color = self._full_trace_forward(network, camera_ray_start, camera_ray_dir,
                                         tf, is_half, toCHW, network_args)

        ctx.raytracing = self
        ctx.network = network
        ctx.camera_ray_start = camera_ray_start
        ctx.camera_ray_dir = camera_ray_dir
        ctx.network_args = network_args
        ctx.tf = tf
        ctx.is_half = is_half
        ctx.toCHW = toCHW
        ctx.save_for_backward(color)

        return color

    @staticmethod
    def backward(ctx, grad_color):
        with torch.no_grad(): # in general, no gradients here
            # unpack saved variables
            color: torch.Tensor
            self: Raytracing = ctx.raytracing
            network: torch.nn.Module = ctx.network
            camera_ray_start: torch.Tensor = ctx.camera_ray_start
            camera_ray_dir: torch.Tensor = ctx.camera_ray_dir
            network_args = ctx.network_args
            tf: pyrenderer.ITransferFunction = ctx.tf
            is_half: bool = ctx.is_half
            toCHW: bool = ctx.toCHW
            color,  = ctx.saved_tensors

            # note: full_trace_forward in forward() converts the tensor from BHWC to BCHW,
            # we have to do the inverse here
            if toCHW:
                color = utils.toHWC(color)
                grad_color = utils.toHWC(grad_color)

            B, H, W, _ = camera_ray_start.shape
            # reshape to (B*H*W)*3
            camera_ray_start = camera_ray_start.view(B * H * W, 3)
            camera_ray_dir = camera_ray_dir.view(B * H * W, 3)

            # find entry and exit times, shape = (3,)
            box_min, box_size, tmin, tmax = \
                self._find_entry_exit(camera_ray_start, camera_ray_dir)

            max_steps = int(torch.max(tmax - tmin).item() / self._stepsize)

            # backward tracing
            assert color.shape == (B, H, W, 4)
            color = color.reshape(B*H*W, 4)
            next_color = color[:, :3]
            next_alpha = color[:, 3:]
            grad_color = grad_color.reshape(B*H*W, 4)
            grad_next_color = grad_color[:, :3]
            grad_next_alpha = grad_color[:, 3:]
            for t in range(max_steps-1, -1, -1):
                tcurrent = tmin + t*self._stepsize
                world_pos = camera_ray_start + tcurrent * camera_ray_dir
                volume_pos = (world_pos - box_min) / box_size
                mask = tcurrent < tmax
                # call network again, with gradients
                with torch.enable_grad():
                    # TODO: pre-integration support
                    if is_half:
                        current_color, prediction = self._predict(volume_pos.half(), camera_ray_dir.half(), network, tf, network_args)
                        current_color = current_color.float()
                    else:
                        current_color, prediction = self._predict(volume_pos, camera_ray_dir, network, tf, network_args)
                # invert blending
                prev_color, prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color = \
                    self._inverse_blend(next_color, next_alpha, current_color, mask,
                                        grad_next_color, grad_next_alpha, stepsize=self._stepsize)
                # accumulate gradients for weights
                torch.autograd.backward(
                    [current_color], # network output
                    [grad_current_color]) # gradient of the network output)
                # next iteration
                next_color = prev_color
                next_alpha = prev_alpha
                grad_next_color = grad_prev_color
                grad_next_alpha = grad_prev_alpha

        return None, None, None, None, None, None, None, None
