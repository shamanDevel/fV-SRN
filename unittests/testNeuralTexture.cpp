#include <catch.hpp>
#include <random>
#include <vector>
#include <Eigen/Core>
#include <torch/torch.h>

#include <neural_textures.cuh>
#include "test_utils.h"

	
TEST_CASE("NeuralTexture-SphereCube", "[math]")
{
	std::default_random_engine rnd(42);
	std::uniform_real_distribution<double> distrUV(-1+1e-3, +1-1e-3);
	std::uniform_int_distribution<int> distrFace(0, 5);

	for (int N=1; N<=100; ++N)
	{
		INFO("N=" << N);
		real2 uv = make_real2(distrUV(rnd), distrUV(rnd));
		int face = distrFace(rnd);

		real3 normal = kernel::EvaluateNeuralTextureUVs<kernel::NeuralTexturesBoundingObject::SphereCube>
			::faceuv2normal(uv, face);

		real2 uvOut; int faceOut;
		kernel::EvaluateNeuralTextureUVs<kernel::NeuralTexturesBoundingObject::SphereCube>
			::normal2faceuv(normal, uvOut, faceOut);

		INFO("normal: " << normal);
		REQUIRE(uv.x == Approx(uvOut.x));
		REQUIRE(uv.y == Approx(uvOut.y));
		REQUIRE(face == faceOut);
	}
}
