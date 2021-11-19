#include <catch.hpp>

#include <string>
#include <vector>
#include <random>
#include <sstream>
#include <tinyformat.h>
namespace fmt = tinyformat;

#include <volume_interpolation_network.h>

using namespace renderer;

class NetworkPytorch
{
	bool hasFourier_ = false;
	int numFourier_ = 0;
	bool hasDirection_ = false;
	bool useDirectionInFourier_ = false;
	torch::Tensor fourierMatrix_;
	torch::Tensor latentGrid_;
	Layer::Activation activation_ = Layer::None;
	float activationParameter_ = 2;
	std::vector<std::shared_ptr<torch::nn::LinearImpl>> hidden_;
	std::shared_ptr<torch::nn::LinearImpl> last_;
	OutputParametrization::OutputMode outputMode_ = OutputParametrization::DENSITY;

public:
	static std::shared_ptr<NetworkPytorch> random(
		bool hasFourier, int numHidden, int hiddenChannels,
		Layer::Activation activation, OutputParametrization::OutputMode outputMode,
		bool hasDirection, bool directionInFourier, int latentChannels, int latentResolution)
	{
		auto n = std::make_shared<NetworkPytorch>();

	    n->hasFourier_ = hasFourier;
		n->hasDirection_ = hasDirection;
		n->useDirectionInFourier_ = directionInFourier;
		int lastChannels = hasDirection ? 6 : 3;
		if (hasFourier)
		{
			n->numFourier_ = (hiddenChannels - (hasDirection ?8:4)) / 2;
			n->fourierMatrix_ = torch::normal(0.0, 1.0, { n->numFourier_, (directionInFourier?6:3) });
			lastChannels += n->numFourier_ * 2;
		}

		if (latentChannels>0)
		{
			n->latentGrid_ = torch::randn({ 1, latentChannels, latentResolution, latentResolution, latentResolution },
				at::TensorOptions().dtype(c10::kFloat));
			lastChannels += latentChannels;
			////Test
			//auto acc = n->latentGrid_.accessor<float, 5>();
			//for (int x=0; x< latentResolution; ++x) for (int y = 0; y < latentResolution; ++y) for (int z = 0; z < latentResolution; ++z)
			//{
			//	acc[0][0][x][y][z] = x < latentResolution/2 ? 0 : 1;
			//	acc[0][1][x][y][z] = y < latentResolution/2 ? 0 : 1;
			//	acc[0][2][x][y][z] = z < latentResolution/2 ? 0 : 1;
			//}
			//for (int c = 0; c < 4; ++c)
			//	std::cout << "Grid [" << c << "]:\n" << n->latentGrid_[0][c] << std::endl;
		}

		n->activation_ = activation;
		n->hidden_.reserve(numHidden);
		for (int i=0; i<numHidden; ++i)
		{
			n->hidden_.push_back(std::make_shared<torch::nn::LinearImpl>(
				torch::nn::LinearOptions(lastChannels, hiddenChannels).bias(true)));
			lastChannels = hiddenChannels;
			////Test
			//if (i==0)
			//{
			//	n->hidden_[0]->bias.detach_().zero_();
			//	at::Tensor& w = n->hidden_[0]->weight;
			//	//std::cout << "Weights 0:\n" << w << std::endl;
			//	w.detach_();
			//	w.zero_();
			//	auto acc = w.accessor<float, 2>();
			//	acc[0][31] = 1;
			//	acc[1][32] = 2;
			//	acc[2][33] = 3;
			//}
		}

		n->outputMode_ = outputMode;
		int outputChannels = OutputParametrization::OutputModeNumChannels[outputMode];
		n->last_ = std::make_shared<torch::nn::LinearImpl>(
			torch::nn::LinearOptions(hiddenChannels, outputChannels).bias(true));

		return n;
	}

	torch::Tensor evaluate(torch::Tensor position, torch::Tensor direction) const
	{
		TORCH_CHECK(position.dim() == 2, "Input expected to be of shape (B,C)");

		////DEBUG
		//std::cout << "Pytorch positions: " << position << std::endl;

		position = position.to(c10::kCUDA, c10::kHalf);
		direction = direction.to(c10::kCUDA, c10::kHalf);

		//fourier
		torch::Tensor y;
		if (hasFourier_)
		{
			auto xBase = hasDirection_
				? torch::cat({ position, direction }, 1)
				: position;
			auto xInput = useDirectionInFourier_
				? torch::cat({ position, direction }, 1)
				: position;
			auto f = torch::matmul(fourierMatrix_, xInput.t()).t();
			auto f2 = 2 * M_PI * f;
			y = torch::cat({ xBase, torch::cos(f2), torch::sin(f2) }, 1);

			////DEBUG
			//std::cout << "Pytorch first layer: " << y[0] << std::endl;
		}
	    else
		{
			y = hasDirection_
				? torch::cat({ position, direction }, 1)
				: position;
		}

		if (latentGrid_.defined())
		{
			//gridPos = position[...,:3].unsqueeze(0).unsqueeze(1).unsqueeze(1)
			auto gridPos = position.unsqueeze(0).unsqueeze(1).unsqueeze(1);
			auto gridPos2 = gridPos * 2 - 1;
			auto output = torch::nn::functional::grid_sample(latentGrid_, gridPos2,
				torch::nn::functional::GridSampleFuncOptions().align_corners(false).padding_mode(torch::kBorder));
			auto latentSpace = output.select(0, 0).select(1, 0).select(1, 0).t();
			y = torch::cat({ y, latentSpace }, 1);

			////Test
			//std::cout << "Pytorch Grid Sampling [0-4]:" << latentSpace.slice(0, 0, 4) << std::endl;
		}

		//hidden
		for (size_t i=0; i<hidden_.size(); ++i)
		{
			y = hidden_[i]->forward(y);
			switch (activation_)
			{
			case Layer::ReLU:
				y = torch::nn::functional::relu(y);
				break;
			case Layer::Sine:
				y = torch::sin(y * activationParameter_);
				break;
			case Layer::Snake: 
			{
				auto tmp = torch::sin(activationParameter_ * y);
				y = y + (1. / activationParameter_) * (tmp * tmp);
			} break;
			case Layer::SnakeAlt:
			{
			    auto t = y + 1 - torch::cos(2 * activationParameter_ * y);
				y = t * (1/(2.f * activationParameter_));
			} break;
			case Layer::Sigmoid:
				y = torch::sigmoid(y);
				break;
			default:
				;//do nothing
			}

			//DEBUG
			//if (i==0)
			//	std::cout << "Pytorch first layer: " << y.slice(0,0,4) << std::endl;
			//else
			//	std::cout << "Pytorch hidden layer " << (hasFourier_?i:int(i)-1) << ": " << y[0] << std::endl;
		}

		//last layer
		y = last_->forward(y);
		y = y.to(c10::kFloat);
		switch (outputMode_)
		{
		case OutputParametrization::DENSITY:
			y = torch::sigmoid(y);
			break;
		case OutputParametrization::DENSITY_DIRECT:
			y = torch::clamp(y, 0, 1);
			break;
		case OutputParametrization::RGBO:
		case OutputParametrization::RGBO_DIRECT:
		{
			auto rgb = y.slice(1, 0, 3);
			auto absorption = y.slice(1, 3, 4);
			if (outputMode_ == OutputParametrization::RGBO)
			{
				rgb = torch::sigmoid(rgb);
				absorption = torch::nn::functional::softplus(absorption);
			} else
			{
				rgb = torch::clamp(rgb, 0, 1);
				absorption = torch::clamp(absorption, 0, {});
			}
			y = torch::cat({ rgb, absorption }, 1);
		}break;
		}

		return y.cpu();
	}

	void toHalfAndCUDA()
	{
		if (hasFourier_)
			fourierMatrix_ = fourierMatrix_.to(c10::kCUDA, c10::kHalf);
		for (size_t i = 0; i < hidden_.size(); ++i)
			hidden_[i]->to(c10::kCUDA, c10::kHalf);
		last_->to(c10::kCUDA, c10::kHalf);
		if (latentGrid_.defined())
			latentGrid_ = latentGrid_.to(c10::kCUDA, c10::kHalf);
	}

	SceneNetwork_ptr toTensorCores(LatentGrid::Encoding encoding = LatentGrid::FLOAT)
	{
		auto n = std::make_shared<SceneNetwork>();
		n->setBoxMin(make_float3(0, 0, 0));
		n->setBoxSize(make_float3(1, 1, 1));

		n->input()->hasDirection = hasDirection_;
		if (hasFourier_)
			n->input()->setFourierMatrixFromTensor(fourierMatrix_, false);
		else
			n->input()->disableFourierFeatures();

		n->output()->outputMode = outputMode_;

		if (latentGrid_.defined())
		{
			auto g = std::make_shared<LatentGridTimeAndEnsemble>(0, 1, 1, 0, 0);
			g->setTimeGridFromTorch(0, latentGrid_, encoding);
			n->setLatentGrid(g);
			n->setTimeAndEnsemble(0, 0);
		}

		for (size_t i = 0; i < hidden_.size(); ++i)
		{
			n->addLayerFromTorch(hidden_[i]->weight, hidden_[i]->bias, activation_, activationParameter_);
		}
		n->addLayerFromTorch(last_->weight, last_->bias, Layer::None);

		return n;
	}
};

struct NetworkInfo
{
	std::shared_ptr<NetworkPytorch> networkPytorch;
	SceneNetwork_ptr networkTensorcores;
	std::string name;
	bool hasColor;
};

static std::vector<NetworkInfo> createNetworks()
{
	std::vector<NetworkInfo> output;

	std::vector<std::pair<int, int>> layersXchannels;
	for (int numHidden : std::vector<int>{ 2,4 })
		for (int hiddenChannels : std::vector<int>{ 32,48 })
			layersXchannels.push_back(std::make_pair(numHidden, hiddenChannels));
	//layersXchannels.insert(layersXchannels.end(),
	//    {
	//	    {20, 32}, { 9, 48 }, { 6, 64 }
	//    });

	std::vector<std::pair<bool, bool>> directionX = {
		{false, false}, {true, false}, {true, true}
	};
	std::vector<std::pair<int, int>> latentX = {
		{0,0}, {16,8}, {16,12}, {32,8}
	};

#if 1
	for (int outputMode=0; outputMode < OutputParametrization::_NUM_OUTPUT_MODES_; ++outputMode)
	for (Layer::Activation activation : std::vector<Layer::Activation>{Layer::ReLU, Layer::Sine, Layer::Snake, Layer::SnakeAlt})
	for (const auto[numHidden, hiddenChannels] : layersXchannels)
	for (bool hasFourier : std::vector<bool>{ false, true })
	for (const auto[hasDirection, directionInFourier] : directionX)
	for (const auto [latentChannels, latentResolution] : latentX)
#else
	bool hasFourier = true;
	int outputMode = 0;
	Layer::Activation activation = Layer::Sine;
	int numHidden = 4;
	int hiddenChannels = 32;
	bool hasDirection = false;
	bool directionInFourier = false;
	int latentChannels = 16;
	int latentResolution = 4; // 32;
#endif
	{
		if (!hasFourier && latentChannels > 0) continue;
		auto networkPytorch = NetworkPytorch::random(
			hasFourier, numHidden, hiddenChannels, activation, 
			OutputParametrization::OutputMode(outputMode),
			hasDirection, directionInFourier,
			latentChannels, latentResolution);
		auto networkTC = networkPytorch->toTensorCores();
		networkPytorch->toHalfAndCUDA();
		
		//generate name
		int numFourier = hasFourier ? (hiddenChannels - (directionInFourier?8:4)) / 2 : 0;
		std::string name = fmt::format("f%d-%s-%s-%d*%d-%s-G%dC%d",
			numFourier, OutputParametrization::OutputModeNames[outputMode],
			Layer::ActivationNames[activation], hiddenChannels, numHidden,
			hasDirection ? (directionInFourier ? "dirF" : "dirD") : "plain",
			    latentResolution, latentChannels);
		//validate
		if (!networkTC->valid())
			FAIL("network invalid: " << name);
		//done
		output.push_back({ networkPytorch, networkTC, name, networkTC->output()->channelsIn()>1 });
	}
	return output;
}

#define MY_INFO(...) INFO(__VA_ARGS__); std::cout << __VA_ARGS__ << std::endl

TEST_CASE("Scene-Representation-Networks", "[modules]")
{
	//generate tensor of positions
	int N = 256;
	torch::Tensor positionsCpu = torch::rand({ N, 3 }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCPU));
	////TEST
 //   auto acc = positionsCpu.accessor<float, 2>();
	//acc[0][0] = 0.1f; acc[0][1] = 0.1f; acc[0][2] = 0.9f;
	//acc[1][0] = 0.9f; acc[1][1] = 0.1f; acc[1][2] = 0.1f;
	//acc[2][0] = 0.1f; acc[2][1] = 0.9f; acc[2][2] = 0.1f;
	//acc[3][0] = 0.9f; acc[3][1] = 0.9f; acc[3][2] = 0.9f;
	torch::Tensor positionsGpu = positionsCpu.to(c10::kCUDA);
	torch::Tensor directionsCpu = torch::rand({ N, 3 }, at::TensorOptions().dtype(c10::kFloat).device(c10::kCPU));
	torch::Tensor directionsGpu = directionsCpu.to(c10::kCUDA);
	CUstream stream = 0;

	//create volume interpolation
	auto volume = std::make_shared<VolumeInterpolationNetwork>();

	//test all networks
	auto networks = createNetworks();
	MY_INFO("Test " << networks.size() << " networks");
	for (size_t i=0; i<networks.size(); ++i)
	{
		MY_INFO("Test network " << (i + 1) << "/" << networks.size() << ": " << networks[i].name);
		auto networkPytorch = networks[i].networkPytorch;
	    auto networkTC = networks[i].networkTensorcores;
		int warpsMixed = networkTC->computeMaxWarps(false);
		int warpsOnlyShared = networkTC->computeMaxWarps(true);
		MY_INFO(" warps mixed: " << warpsMixed << ", only shared: " << warpsOnlyShared);
		if (warpsMixed <= 0 && warpsOnlyShared <= 0) continue;
		//try to render
		torch::Tensor valuesMixed;
		torch::Tensor valuesShared;
		if (warpsMixed > 0) {
			try {
				MY_INFO("  render mixed");
				volume->setOnlySharedMemory(false);
				volume->setNetwork(networkTC);
				valuesMixed = volume->evaluate(positionsGpu, directionsGpu, stream).to(c10::kCPU);
				auto res = cudaDeviceSynchronize();
				if (res != CUDA_SUCCESS)
				{
					FAIL("CUDA error: " << cudaGetErrorString(res));
				}
				//test values (no NAN, correct size)
			} catch (const std::exception& ex)
			{
				FAIL("Unable to render in mixed mode: " << ex.what());
			}
		}
		if (warpsOnlyShared > 0) {
			try {
				MY_INFO("  render shared-only");
			    volume->setOnlySharedMemory(true);
			    volume->setNetwork(networkTC);
			    valuesShared = volume->evaluate(positionsGpu, directionsGpu, stream).to(c10::kCPU);
				auto res = cudaDeviceSynchronize();
				if (res != CUDA_SUCCESS)
				{
					FAIL("CUDA error: " << cudaGetErrorString(res));
				}
			    //test values (no NAN, correct size)
			}
			catch (const std::exception& ex)
			{
				FAIL("Unable to render in shared-only mode: " << ex.what());
			}
		}
		//pytorch reference
		auto valuesPytorch = networkPytorch->evaluate(positionsCpu, directionsCpu);
		//test:
		INFO(valuesMixed);
		INFO(valuesShared);
		INFO(valuesPytorch);
		if (valuesMixed.defined())
		    REQUIRE_FALSE(torch::any(torch::isnan(valuesMixed)).item().toBool());
		if (valuesShared.defined())
			REQUIRE_FALSE(torch::any(torch::isnan(valuesShared)).item().toBool());
		if (valuesMixed.defined() and valuesShared.defined())
			REQUIRE(torch::equal(valuesShared, valuesMixed));
		if (valuesMixed.defined())
			REQUIRE(torch::all(torch::abs(valuesMixed - valuesPytorch) < 1e-2).item().toBool());
		if (valuesShared.defined())
			REQUIRE(torch::all(torch::abs(valuesShared - valuesPytorch) < 1e-2).item().toBool());

		//save and load
		std::stringstream ss(std::iostream::in | std::iostream::out | std::iostream::binary);
		networkTC->save(ss);
		auto networkTCCopy = SceneNetwork::load(ss);
		//compare
		REQUIRE(networkTC->input()->hasDirection == networkTCCopy->input()->hasDirection);
		REQUIRE(networkTC->input()->numFourierFeatures == networkTCCopy->input()->numFourierFeatures);
		REQUIRE(networkTC->input()->fourierMatrix == networkTCCopy->input()->fourierMatrix);
		REQUIRE(networkTC->output()->outputMode == networkTCCopy->output()->outputMode);
		REQUIRE(networkTC->numLayers() == networkTCCopy->numLayers());
		for (int i = 0; i < networkTC->numLayers(); ++i) {
			INFO("hidden: " << i);
			REQUIRE(networkTC->getHidden(i)->channelsIn == networkTCCopy->getHidden(i)->channelsIn);
			REQUIRE(networkTC->getHidden(i)->channelsOut == networkTCCopy->getHidden(i)->channelsOut);
			REQUIRE(networkTC->getHidden(i)->activation == networkTCCopy->getHidden(i)->activation);
			REQUIRE(networkTC->getHidden(i)->weights == networkTCCopy->getHidden(i)->weights);
			REQUIRE(networkTC->getHidden(i)->bias == networkTCCopy->getHidden(i)->bias);
		}
	}
}