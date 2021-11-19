#pragma once

#include "renderer_tensor.cuh"
#include "renderer_utils.cuh"

#include <forward_vector.h>
#include "renderer_cudad_bridge.cuh"
#include "renderer_adjoint.cuh"

namespace kernel
{
	struct CameraReferenceFrameParameters
	{
		real_t fovYRadians;
		real_t aspect; //width / height
		/*
		 * Reference frame matrix, B*3*3.
		 * Where matrix[:,0,:] = eye position, matrix[:,1,:] = right,
		 * matrix[:,2,:] = up
		 */
		Tensor3Read<real_t> matrix;
	};
}

__constant__::kernel::CameraReferenceFrameParameters cameraReferenceFrameParameters;

namespace kernel
{
	typedef pair<real3, real3> Ray_t;
	
	struct CameraReferenceFrame
	{
		__device__ __inline__
		Ray_t eval(const real2& ndc, int batch)
		{
			//fetch parameters
			const auto& fovYRadians = cameraReferenceFrameParameters.fovYRadians;
			const auto& aspect = cameraReferenceFrameParameters.aspect;
			const auto& matrix = cameraReferenceFrameParameters.matrix;
			
			real_t tanFovY = tan(fovYRadians / 2);
			real_t tanFovX = tanFovY * aspect;

			real3 eye = fetchReal3(matrix, batch, 0);
			real3 right = fetchReal3(matrix, batch, 1);
			real3 up = fetchReal3(matrix, batch, 2);
			real3 front = cross(up, right);

			real3 dir = front + ndc.x * tanFovX * right + ndc.y * tanFovY * up;
			dir = normalize(dir);
			return { eye, dir };
		}
	};
}
