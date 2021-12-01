"""
PyTorch-emulation of ray tracing
"""

import numpy as np
import torch

import common.utils as utils
import pyrenderer

class Raytracing:

    def __init__(self,
                 settings,# : pyrenderer.RendererInputs,
                 fov_y_radians: float,
                 network_output : str,
                 stepsize: float):
        """

        :param settings: the renderer settings (screen size, box size)
        :param fov_y_radians: the fov for the camera
        :param network_output: the network output mode, either 'color' or 'density'
        :param stepsize: the stepsize in world space
        """
        assert network_output in ["color", "density"]
        self._settings = settings
        self._fov_y_radians = fov_y_radians
        self._network_output = network_output
        self.stepsize = stepsize

    @staticmethod
    def intersection_aabb(
            ray_start, ray_dir, box_min, box_size, return_face = False):
        """
        Computes AABB-intersection. All tensors are of shape (B,3)
        with broadcasting over the batches.

        The face indices are:
         0,1: X
         2,3: Y
         4,5: Z

        :return: (tmin, tmax) of shape (B,1) OR (tmin, tmax, face_index) if return_face=True
        """

        inv_ray_dir = 1.0 / ray_dir
        if return_face:
            # http://www.jcgt.org/published/0007/03/04/paper-lowres.pdf
            box_radius = 0.5 * box_size
            box_center = box_min + box_radius
            inv_box_radius = 1.0 / box_radius
            ray_origin = ray_start - box_center
            winding = torch.max(torch.abs(ray_origin)*inv_box_radius, dim=1, keepdim=True)[0]
            winding = torch.sign(winding-1)
            sgn = -torch.sign(ray_dir)
            # distance to plane
            d = (box_radius * winding * sgn - ray_origin) * inv_ray_dir
            def test(u, v, w):
                b1 = torch.abs(ray_origin[:, v:v + 1] + ray_dir[:, v:v + 1] * d[:, u:u + 1]) < box_radius[:, v:v + 1]
                b2 = torch.abs(ray_origin[:, w:w + 1] + ray_dir[:, w:w + 1] * d[:, u:u + 1]) < box_radius[:, w:w + 1]
                return torch.logical_and(b1, b2)
            testX = test(0, 1, 2); testY = test(1, 2, 0); testZ = test(2, 0, 1)

            # face index
            sgn2 = (ray_dir<0) * 1.0
            face = torch.where(
                testX, sgn2[:,0:1],
                torch.where(
                    testY, 2+sgn2[:,1:2], 4+sgn2[:,2:3]
                ))

            # distance
            t135 = (box_min - ray_start) * inv_ray_dir
            t246 = (box_min + box_size - ray_start) * inv_ray_dir
            tmin = torch.max(torch.minimum(t135, t246), dim=1, keepdim=True)[0]
            tmax = torch.min(torch.maximum(t135, t246), dim=1, keepdim=True)[0]
            return tmin, tmax, face
        else:
            t135 = (box_min - ray_start) * inv_ray_dir
            t246 = (box_min + box_size - ray_start) * inv_ray_dir
            tmin = torch.max(torch.minimum(t135, t246), dim=1, keepdim=True)[0]
            tmax = torch.min(torch.maximum(t135, t246), dim=1, keepdim=True)[0]
            return tmin, tmax

    def _generate_camera_ray(self, camera_viewport: torch.Tensor):
        # output is in B*H*W*3
        camera_ray_start, camera_ray_dir = pyrenderer.Camera.generate_rays(
            camera_viewport, self._fov_y_radians,
            self._settings.screen_size.x, self._settings.screen_size.y)
        B, H, W, _ = camera_ray_start.shape
        # reshape to (B*H*W)*3
        camera_ray_start = camera_ray_start.view(B * H * W, 3)
        camera_ray_dir = camera_ray_dir.view(B * H * W, 3)
        return B, H, W, camera_ray_start, camera_ray_dir

    def _find_entry_exit(self, camera_ray_start, camera_ray_dir):
        box_min = torch.from_numpy(utils.cvector_to_numpy(self._settings.box_min))
        box_size = torch.from_numpy(utils.cvector_to_numpy(self._settings.box_size))
        box_min = box_min.unsqueeze(0).to(device=camera_ray_start.device, dtype=camera_ray_start.dtype)
        box_size = box_size.unsqueeze(0).to(device=camera_ray_start.device, dtype=camera_ray_start.dtype)
        tmin, tmax = Raytracing.intersection_aabb(
            camera_ray_start, camera_ray_dir, box_min, box_size)
        return box_min, box_size, tmin, tmax

    def _predict(self, volume_pos: torch.Tensor,
                  network: torch.nn.Module,
                  tf: torch.Tensor = None,
                  tf_mode = None):
        prediction = network(volume_pos)
        BHW = prediction.shape[0]
        if self._network_output == "color":
            assert prediction.shape == (BHW, 4), "wrong output shape %s" % prediction.shape
            color = prediction
        elif self._network_output == "density":
            assert prediction.shape == (BHW, 1), "wrong output shape %s" % prediction.shape
            assert (tf is not None) and (tf_mode is not None), \
                "if the network predictions densities, the TF must be specified"
            color = pyrenderer.TFUtils.preshade_volume(
                prediction.t().unsqueeze(-1).unsqueeze(-1),
                tf, tf_mode)[:, :, 0, 0].t()
        else:
            raise ValueError("Unknown network prediction mode")
        return color

    def _blend(self, prev_color, prev_alpha, current_color, mask):
        current_alpha = 1 - torch.exp(-current_color[:, 3:] * self.stepsize)
        zeros = torch.zeros(1,1, dtype=current_alpha.dtype, device=current_alpha.device)
        current_alpha = torch.where(mask, current_alpha, zeros)  # out of bounds
        next_color = prev_color + (1 - prev_alpha) * current_color[:, :3] * current_alpha
        next_alpha = prev_alpha + (1 - prev_alpha) * current_alpha
        return next_color, next_alpha

    def _inverse_blend(self,
                      next_color, next_alpha, current_color, mask,
                      grad_next_color, grad_next_alpha):

        current_rgb = current_color[:, :3]
        current_alpha = 1 - torch.exp(-current_color[:, 3:] * self.stepsize)
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
        grad_current_alpha = grad_current_alpha * self.stepsize * torch.exp(-current_color[:, 3:] * self.stepsize)

        grad_current_color = torch.cat((grad_current_rgb, grad_current_alpha), dim=1)
        return prev_color, prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color

    def full_trace_forward(self,
                           network: torch.nn.Module,
                           camera_viewport: torch.Tensor,
                           tf: torch.Tensor = None,
                           tf_mode = None,
                           is_half = False):
        # generate camera rays
        B, H, W, camera_ray_start, camera_ray_dir = \
            self._generate_camera_ray(camera_viewport)

        # find entry and exit times, shape = (3,)
        box_min, box_size, tmin, tmax = \
            self._find_entry_exit(camera_ray_start, camera_ray_dir)

        max_steps = int(torch.max(tmax-tmin).item() / self.stepsize)
        #print("Raytrace with", max_steps, "steps")

        # perform stepping
        renderer_dtype_torch = camera_viewport.dtype
        final_color = torch.zeros((B * H * W, 3), dtype=renderer_dtype_torch,
                                  device=camera_viewport.device)
        final_alpha = torch.zeros((B * H * W, 1), dtype=renderer_dtype_torch,
                                  device=camera_viewport.device)
        for t in range(max_steps):
            tcurrent = tmin + t*self.stepsize
            world_pos = camera_ray_start + tcurrent * camera_ray_dir
            volume_pos = (world_pos - box_min) / box_size
            # call network
            if is_half:
                color = self._predict(volume_pos.half(), network, tf, tf_mode).float()
            else:
                color = self._predict(volume_pos, network, tf, tf_mode)
            # blending
            final_color, final_alpha = self._blend(
                final_color, final_alpha, color, tcurrent < tmax)

        # done, reshape back to original
        rgba = torch.cat((final_color, final_alpha), dim=1)
        return rgba.view(B, H, W, 4)

    def checkpointed_trace(self,
                           network: torch.nn.Module,
                           camera_viewport: torch.Tensor,
                           tf: torch.Tensor = None,
                           tf_mode = None):
        """
        Same syntax and semantic as self.full_trace_forward,
        but computes the gradient step by step via local inversion.
        This is a bit more costly in terms of performance, but does not require additional memory
        """
        # ugly hack, but without an input with requires_grad=True,
        # the autograd.Function is not recorded
        camera_viewport0 = camera_viewport.clone().requires_grad_()
        out = _CheckpointedTrace.apply(self, network, camera_viewport0, tf, tf_mode)
        return out

class _CheckpointedTrace(torch.autograd.Function):
    @staticmethod
    def forward(ctx, self: Raytracing,
                network: torch.nn.Module,
                camera_viewport: torch.Tensor,
                tf: torch.Tensor = None,
                tf_mode = None):
        color = self.full_trace_forward(network, camera_viewport, tf, tf_mode)

        ctx.raytracing = self
        ctx.network = network
        ctx.camera_viewport = camera_viewport
        ctx.tf = tf
        ctx.tf_mode = tf_mode
        ctx.save_for_backward(color)

        return color

    @staticmethod
    def backward(ctx, grad_color):
        with torch.no_grad(): # in general, no gradients here
            # unpack saved variables
            color: torch.Tensor
            self: Raytracing = ctx.raytracing
            network: torch.nn.Module = ctx.network
            camera_viewport: torch.Tensor = ctx.camera_viewport
            tf: torch.Tensor = ctx.tf
            tf_mode: pyrenderer.TFMode = ctx.tf_mode
            #color, self, network, camera_viewport, tf, tf_mode = ctx.saved_tensors
            color,  = ctx.saved_tensors

            # generate camera rays
            B, H, W, camera_ray_start, camera_ray_dir = \
                self._generate_camera_ray(camera_viewport)

            # find entry and exit times, shape = (3,)
            box_min, box_size, tmin, tmax = \
                self._find_entry_exit(camera_ray_start, camera_ray_dir)

            max_steps = int(torch.max(tmax - tmin).item() / self.stepsize)

            # backward tracing
            color = color.view(B*H*W, 4)
            next_color = color[:, :3]
            next_alpha = color[:, 3:]
            grad_color = grad_color.view(B*H*W, 4)
            grad_next_color = grad_color[:, :3]
            grad_next_alpha = grad_color[:, 3:]
            for t in range(max_steps-1, -1, -1):
                tcurrent = tmin + t*self.stepsize
                world_pos = camera_ray_start + tcurrent * camera_ray_dir
                volume_pos = (world_pos - box_min) / box_size
                mask = tcurrent < tmax
                # call network again, with gradients
                with torch.enable_grad():
                    current_color = self._predict(volume_pos, network, tf, tf_mode)
                # invert blending
                prev_color, prev_alpha, grad_prev_color, grad_prev_alpha, grad_current_color = \
                    self._inverse_blend(next_color, next_alpha, current_color, mask,
                                        grad_next_color, grad_next_alpha)
                # accumulate gradients for weights
                torch.autograd.backward(
                    [current_color], # network output
                    [grad_current_color]) # gradient of the network output)
                # next iteration
                next_color = prev_color
                next_alpha = prev_alpha
                grad_next_color = grad_prev_color
                grad_next_alpha = grad_prev_alpha

        return None, None, None, None, None
