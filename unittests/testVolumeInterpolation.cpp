#include <catch.hpp>
#include <random>
#include <cuda_runtime.h>
#include <filesystem>

#include "test_utils.h"

#include <volume_interpolation_grid.h>
#include <c10/cuda/CUDAStream.h>

TEST_CASE("VolumeInterpolation-Evaluate", "[modules]")
{
	//load volume
	auto root_path = std::filesystem::path(__FILE__).parent_path().parent_path();
	auto volume_path = root_path / "example-volume.cvol";
	INFO("Load " << volume_path);
	renderer::Volume_ptr volume = std::make_shared<renderer::Volume>(volume_path.string());

	//create interpolation
	auto interpolation = std::make_shared<renderer::VolumeInterpolationGrid>();
	interpolation->setSource(volume, 0);

	//create position tensor
	int N = 1024;
	torch::Tensor positions = torch::rand({ N,3 }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCUDA));
	torch::Tensor values = interpolation->evaluate(
		positions, {}, c10::cuda::getCurrentCUDAStream());
	REQUIRE(values.dim() == 2);
	REQUIRE(values.size(0) == N);
	REQUIRE(values.size(1) == 1);

	////print
	//torch::Tensor positionsCpu = positions.cpu();
	//torch::Tensor valuesCpu = values.cpu();
	//const auto positionsAcc = positionsCpu.accessor<float, 2>();
	//const auto valuesAcc = valuesCpu.accessor<float, 2>();
	//for (int i=0; i<N; ++i)
	//{
	//	std::cout << "(" << positionsAcc[i][0] << ", " << positionsAcc[i][1] << ", " <<
	//		positionsAcc[i][2] << ") -> " << valuesAcc[i][0] << "\n";
	//}
	//std::cout << std::flush;
}