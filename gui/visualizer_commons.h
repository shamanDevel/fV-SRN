#pragma once

/*
 * DEPRECATED
 */

#include <cuMat/src/ForwardDeclarations.h>

enum RenderMode
{
	IsosurfaceRendering,
	DirectVolumeRendering
};

typedef cuMat::Matrix<float, cuMat::Dynamic, cuMat::Dynamic, 2, cuMat::ColumnMajor> FlowTensor;