#include "volume_interpolation_grid.h"

#include <sstream>
#include <portable-file-dialogs.h>
#include <magic_enum.hpp>
#include <cuMat/src/Errors.h>

#include "helper_math.cuh"
#include "IconsFontAwesome5.h"
#include "imgui_internal.h"
#include "kernel_loader.h"
#include "renderer_tensor.cuh"
#include "pytorch_utils.h"
#include "renderer_commons.cuh"


const std::string renderer::VolumeInterpolationGrid::VolumeInterpolationNames[] = {
	"Nearest", "Linear", "Cubic"
};

const std::string renderer::VolumeInterpolationGrid::Feature2DensityNames[] = {
	/* Identity          */ "",
	/* VelocityX         */ " - X",
	/* VelocityY         */ " - Y",
	/* VelocityZ         */ " - Z",
	/* VelocityMagnitude */ " - mag",
	/* Density           */ " - density",
	/* DensityCurvature  */ " - density+curvature"
};
//The datatype of the 3D texture / texture sampler in renderer_volume_grid
const std::string renderer::VolumeInterpolationGrid::Feature2DensityTextureType[] = {
	/* Identity          */ "float",
	/* VelocityX         */ "float4",
	/* VelocityY         */ "float4",
	/* VelocityZ         */ "float4",
	/* VelocityMagnitude */ "float4",
	/* Density           */ "float4",
	/* DensityCurvature  */ "float4"
};
//The name of the function to extract the scalar data
const std::string renderer::VolumeInterpolationGrid::Feature2DensityTextureChannel[] = {
	/* Identity          */ "getDirect",
	/* VelocityX         */ "getX",
	/* VelocityY         */ "getY",
	/* VelocityZ         */ "getZ",
	/* VelocityMagnitude */ "getMagnitude",
	/* Density           */ "getW",
	/* DensityCurvature  */ "getColor" //I want all
};

renderer::VolumeInterpolationGrid::VolumeInterpolationGrid()
	: IVolumeInterpolation(true)
   , source_(VolumeSource::EMPTY)
   , interpolation_(VolumeInterpolation::NEAREST_NEIGHBOR)
   , minDensity_(0)
   , maxDensity_(1)
   , mipmapLevel_(0)
   , selectedDensityFeatureIndex_(0)
   , selectedVelocityFeatureIndex_(0)
   , selectedColorFeatureIndex_(0)
   , currentEnsemble_(0)
   , currentTimestep_(0)
   , histogramCache_(100, histogramHash)
   , histogramStream_(0)
   , histogramCompletionEvent_(0)
   , histogramExtractionRunning_(false)
   , histogramDevice_(nullptr)
   , gridResolutionNewBehavior_(false)
{
	CUMAT_SAFE_CALL(cudaStreamCreate(&histogramStream_));
	CUMAT_SAFE_CALL(cudaEventCreateWithFlags(
		&histogramCompletionEvent_, cudaEventBlockingSync | cudaEventDisableTiming));
	CUMAT_SAFE_CALL(cudaMalloc(&histogramDevice_, sizeof(kernel::VolumeHistogram)));
}

renderer::VolumeInterpolationGrid::~VolumeInterpolationGrid()
{
	CUMAT_SAFE_CALL_NO_THROW(cudaStreamDestroy(histogramStream_));
	CUMAT_SAFE_CALL_NO_THROW(cudaEventDestroy(histogramCompletionEvent_));
	CUMAT_SAFE_CALL_NO_THROW(cudaFree(histogramDevice_));
}

bool renderer::VolumeInterpolationGrid::extractDensityFeaturesFromVolume()
{
	FeatureDescriptorDensity oldFeature{ -1, Feature2Density::Identity };
	if (selectedDensityFeatureIndex_ >= 0 & selectedDensityFeatureIndex_ < availableDensityFeatures_.size())
	{
		oldFeature = availableDensityFeatures_[selectedDensityFeatureIndex_];
	}

	availableDensityFeatures_.clear();
	selectedDensityFeatureIndex_ = 0;
	for (int i=0; i<volume_->numFeatures(); ++i)
	{
		const auto& f = volume_->getFeature(i);
		if (f->numChannels() == 1)
		{
		    //density
			availableDensityFeatures_.push_back({ i, Feature2Density::Identity });
		} else if (f->numChannels() == 4)
		{
		    //velocities + density in w-channels
			availableDensityFeatures_.push_back({ i, Feature2Density::Density });
			availableDensityFeatures_.push_back({ i, Feature2Density::VelocityX });
			availableDensityFeatures_.push_back({ i, Feature2Density::VelocityY });
			availableDensityFeatures_.push_back({ i, Feature2Density::VelocityZ });
			availableDensityFeatures_.push_back({ i, Feature2Density::VelocityMagnitude });
			availableDensityFeatures_.push_back({ i, Feature2Density::DensityCurvature });
		}
		else
		{
		    //unknown -> ignore
			std::cout << "Unknown feature channel '" << f->name() << "' with " << f->numChannels() << " channels" << std::endl;
		}
	}

	if (availableDensityFeatures_.empty())
	{
		std::cerr << "Volume does not contain compatible features!" << std::endl;
		return false;
	}

	//search if we have a match with one of the new features
	for (int i=0; i<availableDensityFeatures_.size(); ++i)
	{
	    if (availableDensityFeatures_[i].featureIndex == oldFeature.featureIndex &&
			availableDensityFeatures_[i].mapping == oldFeature.mapping)
	    {
			selectedDensityFeatureIndex_ = i;
			break;
	    }
	}
	return true;
}

void renderer::VolumeInterpolationGrid::setSource(Volume_ptr v, int mipmap)
{
	static int previousDensityFeatureIndex = -1;
	source_ = VolumeSource::VOLUME;
	bool changed = volume_ != v;
	volume_ = v;
	mipmapLevel_ = mipmap;
	tensor_ = Parameter<torch::Tensor>();

	//density features
	if (v == nullptr)
		return;
	if (!extractDensityFeaturesFromVolume()) {
		return;
	}
	const auto& m = availableDensityFeatures_[selectedDensityFeatureIndex_];
	const auto& feature = volume_->getFeature(m.featureIndex);
	if (m.featureIndex != previousDensityFeatureIndex)
	{
		previousDensityFeatureIndex = m.featureIndex;
		changed = true;
	}

	//velocity + color features
	int previousVelocityFeature =
		(selectedVelocityFeatureIndex_ >= 0 && selectedVelocityFeatureIndex_ < availableVelocityFeatures_.size())
		? availableVelocityFeatures_[selectedVelocityFeatureIndex_].featureIndex
		: -1;
	int previousColorFeature =
		(selectedColorFeatureIndex_ >= 0 && selectedColorFeatureIndex_ < availableColorFeatures_.size())
		? availableColorFeatures_[selectedColorFeatureIndex_].featureIndex
		: -1;
	selectedVelocityFeatureIndex_ = 0;
	selectedColorFeatureIndex_ = 0;
	availableVelocityFeatures_.clear();
	availableColorFeatures_.clear();
	for (int i = 0; i < volume_->numFeatures(); ++i)
	{
		const auto& f = volume_->getFeature(i);
		if (f->numChannels() == 3 || f->numChannels()==4)
		{
			if (i == previousVelocityFeature)
				selectedVelocityFeatureIndex_ = availableVelocityFeatures_.size();
			availableVelocityFeatures_.push_back({i});
		}
		if (f->numChannels() == 4)
		{
			if (i == previousColorFeature)
				selectedColorFeatureIndex_ = availableColorFeatures_.size();
			availableColorFeatures_.push_back({i});
		}
	}

	if (feature->getLevel(mipmap) == nullptr)
		feature->createMipmapLevel(mipmap, Volume::MipmapFilterMode::AVERAGE);
    feature->getLevel(mipmap)->copyCpuToGpu();

	double3 worldSize = make_double3(
		v->worldSizeX(), v->worldSizeY(), v->worldSizeZ()
	);
	setObjectResolution(feature->getLevel(mipmap)->size());
	setBoxMin(-worldSize / 2.0);
	setBoxMax( worldSize / 2.0);
}

void renderer::VolumeInterpolationGrid::setSource(const torch::Tensor& t)
{
	CHECK_DIM(t, 4);
	TORCH_CHECK((t.dtype() == c10::kFloat || t.dtype() == c10::kDouble),
		"tensor must be of type float or double, but is ", t.dtype());
	
	source_ = VolumeSource::TORCH_TENSOR;
	volume_ = nullptr;
	ensembleFactory_ = nullptr;
	mipmapLevel_ = 0;
	tensor_.value = t;
	tensor_.grad = torch::Tensor();
	tensor_.forwardIndex = torch::Tensor();

	minDensity_ = torch::min(t).item().toFloat();
	maxDensity_ = torch::max(t).item().toFloat();

	int3 res = make_int3(t.size(1), t.size(2), t.size(3));
	setObjectResolution(res);
	double voxelSize = 1.0 / std::max({ res.x, res.y, res.z });
	double3 worldSize = make_double3(res) * voxelSize;
	setBoxMin(-worldSize / 2.0);
	setBoxMax(worldSize / 2.0);
}

void renderer::VolumeInterpolationGrid::setSource(VolumeEnsembleFactory_ptr factory)
{
	TORCH_CHECK(factory != nullptr);
	ensembleFactory_ = factory;
	setEnsembleAndTime(0, 0, mipmapLevel_);
}

void renderer::VolumeInterpolationGrid::setEnsembleAndTime(int ensemble, int time, int mipmap)
{
	if ((currentEnsemble_ != ensemble || currentTimestep_ != time) && volume_!=nullptr)
	{
	    //unload GPU resources to save memory
		volume_->clearGpuResources();
	}
	currentEnsemble_ = ensemble;
	currentTimestep_ = time;
	setSource(ensembleFactory_->loadVolume(ensemble, time), mipmap);
}

renderer::Volume_ptr renderer::VolumeInterpolationGrid::volume() const
{
	TORCH_CHECK(source_ == VolumeSource::VOLUME,
		"source()==VOLUME required, but is currently ",
		magic_enum::enum_name(source_));
	return volume_;
}

int renderer::VolumeInterpolationGrid::mipmapLevel() const
{
	TORCH_CHECK(source_ == VolumeSource::VOLUME,
		"source()==VOLUME required, but is currently ",
		magic_enum::enum_name(source_));
	return mipmapLevel_;
}

renderer::Parameter<at::Tensor> renderer::VolumeInterpolationGrid::tensor() const
{
	TORCH_CHECK(source_ == VolumeSource::TORCH_TENSOR,
	            "source()==TORCH_TENSOR required, but is currently ", 
				magic_enum::enum_name(source_));
	return tensor_;
}

renderer::VolumeEnsembleFactory_ptr renderer::VolumeInterpolationGrid::ensembleFactory() const
{
	TORCH_CHECK(source_ == VolumeSource::VOLUME,
		"source()==VOLUME required, but is currently ",
		magic_enum::enum_name(source_));
	return ensembleFactory_;
}

int renderer::VolumeInterpolationGrid::currentEnsemble() const
{
	TORCH_CHECK(source_ == VolumeSource::VOLUME,
		"source()==VOLUME required, but is currently ",
		magic_enum::enum_name(source_));
	return currentEnsemble_;
}

int renderer::VolumeInterpolationGrid::currentTimestep() const
{
	TORCH_CHECK(source_ == VolumeSource::VOLUME,
		"source()==VOLUME required, but is currently ",
		magic_enum::enum_name(source_));
	return currentTimestep_;
}

std::string renderer::VolumeInterpolationGrid::getName() const
{
	return "Grid";
}

const std::string renderer::VolumeInterpolationGrid::UI_KEY_HISTOGRAM = "volume##histogram";
const std::string renderer::VolumeInterpolationGrid::UI_KEY_MIN_DENSITY = "volume##minDensity";
const std::string renderer::VolumeInterpolationGrid::UI_KEY_MAX_DENSITY = "volume##maxDensity";

static bool SliderIntWithButtons(const char* label, int* v, int v_min, int v_max, const char* format = "%d")
{
	bool changed = false;

	ImGuiStyle& style = ImGui::GetStyle();
	float w = ImGui::CalcItemWidth();
	float spacing = style.ItemInnerSpacing.x;
	float button_sz = ImGui::GetFrameHeight();
	ImGui::PushItemWidth(w - spacing * 2.0f - button_sz * 2.0f);

	const std::string s0 = std::string("##") + label;
	if (ImGui::SliderInt(s0.c_str(), v, v_min, v_max, format))
		changed = true;

	ImGui::PopItemWidth();
	ImGui::SameLine(0, spacing);
	const std::string s1 = std::string("##l##") + label;
	if (ImGui::ArrowButton(s1.c_str(), ImGuiDir_Left))
	{
		if (*v > v_min)
		{
			--*v;
			changed = true;
		}
	}
	ImGui::SameLine(0, spacing);
	const std::string s2 = std::string("##r##") + label;
	if (ImGui::ArrowButton(s2.c_str(), ImGuiDir_Right))
	{
		if (*v < v_max)
		{
			++*v;
			changed = true;
		}
	}
	ImGui::SameLine(0, style.ItemInnerSpacing.x);
	const std::string s3 = std::string(label).substr(0, std::string(label).find('#'));
	ImGui::Text(s3.c_str());

	return changed;
}

bool renderer::VolumeInterpolationGrid::drawUI(UIStorage_t& storage)
{
	setGridResolutionNewBehavior(true); //fix to new behavior in UI
	bool changed = false;
	
	if (source_ == VolumeSource::TORCH_TENSOR)
	{
		ImGui::TextColored(ImVec4(1, 0, 0, 1),
			"Volume specified by PyTorch Tensor,\ncan't edit in the UI");
		return false;
	}
	auto filename = volumeFullFilename_.filename().string();
	ImGui::InputText("##VolumeInterpolationGridInput", 
		filename.data(), 
		filename.size() + 1, 
		ImGuiInputTextFlags_ReadOnly);
	ImGui::SameLine();
	if (ImGui::Button(ICON_FA_FOLDER_OPEN "##VolumeInterpolationGrid"))
	{
		loadVolumeDialog();
		changed = true;
	}
	ImGui::SameLine();
	if (ImGui::ButtonEx("Reload##VolumeInterpolationGrid", ImVec2(0,0), 
		source_==VolumeSource::VOLUME && !volumeFullFilename_.empty() ? 0 : ImGuiButtonFlags_Disabled))
	{
		loadVolumeFromPath(volumeFullFilename_.string());
		changed = true;
	}

	if (source_ == VolumeSource::EMPTY)
	{
		ImGui::TextColored(ImVec4(1, 0, 0, 1),
			"No volume loaded");
		return changed;
	}

	const auto f = ensembleFactory_;
	if (f != nullptr)
	{
		int ensemble = currentEnsemble_;
		int timestep = currentTimestep_;
	    if (SliderIntWithButtons("Ensemble##VolumeInterpolationGridInput", &ensemble,
			0, f->numEnsembles()-1))
	    {
			setEnsembleAndTime(ensemble, timestep, mipmapLevel());
			changed = true;
	    }
		if (SliderIntWithButtons("Timestep##VolumeInterpolationGridInput", &timestep,
			0, f->numTimesteps() - 1))
		{
			setEnsembleAndTime(ensemble, timestep, mipmapLevel());
			changed = true;
		}
	}

	if (availableDensityFeatures_.empty())
	{
		ImGui::TextColored(ImVec4(1, 0, 0, 1), "Volume does not contain scalar or\nscalar-compatible features!");
		return changed;
	}

	//features
	const auto getFeatureName = [this](int index)
	{
		const auto& d = availableDensityFeatures_[index];
		const auto& f = volume_->getFeature(d.featureIndex);
		return f->name() + Feature2DensityNames[int(d.mapping)];
	};
	const std::string currentDensityFeatureName = getFeatureName(selectedDensityFeatureIndex_);
	if (ImGui::BeginCombo("Density Feature##VolumeInterpolationGridInput", currentDensityFeatureName.c_str()))
	{
		for (size_t i=0; i<availableDensityFeatures_.size(); ++i)
		{
			bool is_selected = (i == selectedDensityFeatureIndex_);
			const std::string label = getFeatureName(i);
			if (ImGui::Selectable(label.c_str(), is_selected))
			{
				selectedDensityFeatureIndex_ = i;
				changed = true;
			}
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	const std::string currentVelocityFeatureName = 
		availableVelocityFeatures_.empty()
        ? "<Not available>"
        : volume_->getFeature(availableVelocityFeatures_[selectedVelocityFeatureIndex_].featureIndex)->name();
	if (ImGui::BeginCombo("Velocity Feature##VolumeInterpolationGridInput", currentVelocityFeatureName.c_str()))
	{
		for (size_t i = 0; i < availableVelocityFeatures_.size(); ++i)
		{
			bool is_selected = (i == selectedVelocityFeatureIndex_);
			const std::string label = volume_->getFeature(availableVelocityFeatures_[i].featureIndex)->name();
			if (ImGui::Selectable(label.c_str(), is_selected))
			{
				selectedVelocityFeatureIndex_ = i;
				changed = true;
			}
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	const std::string currentColorFeatureName =
		availableColorFeatures_.empty()
		? "<Not available>"
		: volume_->getFeature(availableColorFeatures_[selectedColorFeatureIndex_].featureIndex)->name();
	if (ImGui::BeginCombo("Color Feature##VolumeInterpolationGridInput", currentColorFeatureName.c_str()))
	{
		for (size_t i = 0; i < availableColorFeatures_.size(); ++i)
		{
			bool is_selected = (i == selectedColorFeatureIndex_);
			const std::string label = volume_->getFeature(availableColorFeatures_[i].featureIndex)->name();
			if (ImGui::Selectable(label.c_str(), is_selected))
			{
				selectedColorFeatureIndex_ = i;
				changed = true;
			}
			if (is_selected)
				ImGui::SetItemDefaultFocus();
		}
		ImGui::EndCombo();
	}

	FeatureDescriptorDensity featureDescriptor = availableDensityFeatures_[selectedDensityFeatureIndex_];
	Volume::Feature_ptr feature = volume_->getFeature(featureDescriptor.featureIndex);

	static const std::vector<std::pair<int, std::string>> MipLevelNames = {
		{0, "orig"}, {1, "2x"}, {2, "4x"}, {3, "8x"}
	};
	for (int i=0; i<MipLevelNames.size(); ++i)
	{
		if (i > 0) ImGui::SameLine();
		if (ImGui::RadioButton(MipLevelNames[i].second.c_str(), mipmapLevel_==MipLevelNames[i].first))
		{
			mipmapLevel_ = MipLevelNames[i].first;
			feature->createMipmapLevel(mipmapLevel_, Volume::MipmapFilterMode::AVERAGE);
			setSource(volume_, mipmapLevel_);
			changed = true;
		}
	}

	if (ImGui::SliderInt("Interpolation##VolumeInterpolationGrid",
		reinterpret_cast<int*>(&interpolation_),
		0, static_cast<int>(VolumeInterpolation::_COUNT_)-1,
		VolumeInterpolationNames[static_cast<int>(interpolation_)].c_str()))
	{
		changed = true;
	}

	//histogram
	HistogramValue_ptr histogram = requestHistogram();
	if (histogram)
	{
		minDensity_ = histogram->minDensity;
		maxDensity_ = histogram->maxDensity;
	}
	ImGui::Text("Resolution: %d, %d, %d\nDensity: min=%.3f, max=%.3f",
		objectResolution().x, objectResolution().y, objectResolution().z,
		minDensity(), maxDensity());

	//UI Storage

	storage[UI_KEY_HISTOGRAM] = histogram;
	storage[UI_KEY_MIN_DENSITY] = static_cast<float>(minDensity());
	storage[UI_KEY_MAX_DENSITY] = static_cast<float>(maxDensity());
	
	if (backgroundGui_)
		backgroundGui_();

	if (newVolumeLoaded_)
	{
		newVolumeLoaded_ = false;
		changed = true;
	}
	return changed;
}

renderer::VolumeInterpolationGrid::HistogramValue_ptr renderer::VolumeInterpolationGrid::requestHistogram()
{
	HistogramKey key;
	key.volumePtr = volume_.get();
	key.featureIndex = availableDensityFeatures_[selectedDensityFeatureIndex_].featureIndex;
	key.mipmapLevel = mipmapLevel_;
	key.mapping = availableDensityFeatures_[selectedDensityFeatureIndex_].mapping;

	if (histogramCache_.exist(key))
	{
		return histogramCache_.get(key);
	}

	if (histogramExtractionRunning_) {
		//already an extraction is running
		//check if done
		auto status = cudaEventQuery(histogramCompletionEvent_);
		if (status == cudaSuccess)
		{
		    //we are done, emplace in cache
			histogramExtractionRunning_ = false;
			HistogramValue_ptr histo = std::make_shared<kernel::VolumeHistogram>(histogramHost_);
			histogramCache_.put(histogramCurrentKey_, histo);
			if (histogramCurrentKey_ == key)
			{
			    //we are still at the same key, return
				return histo;
			}
			//key has changed, we have to start the extraction again -> below
		}
		else if (status == cudaErrorNotReady)
		{
		    //not done yet
			return nullptr;
		} else
		{
		    //error!
			printError(static_cast<CUresult>(status), "Error while waiting for histogram extraction kernel");
			return nullptr;
		}
	}

	//start extraction

	//clear histogram
	kernel::VolumeHistogram* histogramOutDevice = histogramDevice_;
	CUMAT_SAFE_CALL(cudaMemsetAsync(histogramOutDevice, 0, sizeof(kernel::VolumeHistogram), histogramStream_));

	//compile kernel
	GlobalSettings s{};
	s.scalarType = GlobalSettings::kFloat;
	s.volumeShouldProvideNormals = false;
	s.interpolationInObjectSpace = true;
	this->prepareRendering(s);
	const std::string kernelName = "HistogramExtractKernel";
	std::vector<std::string> constantNames;
	if (const auto c = getConstantDeclarationName(s); !c.empty())
		constantNames.push_back(c);
	std::stringstream extraSource;
	extraSource << "#define KERNEL_DOUBLE_PRECISION 0\n";
	extraSource << "#define KERNEL_SYNCHRONIZED_TRACING 0\n";
	extraSource << getDefines(s) << "\n";
	fillExtraSourceCode(s, extraSource);
	for (const auto& i : getIncludeFileNames(s))
		extraSource << "\n#include \"" << i << "\"\n";
	extraSource << "#define VOLUME_INTERPOLATION_T " <<
		getPerThreadType(s) << "\n";
	extraSource << "#include \"renderer_volume_kernels6.cuh\"\n";
	const auto fun0 = KernelLoader::Instance().getKernelFunction(
		kernelName, extraSource.str(), constantNames, false, false);
	if (!fun0.has_value())
		throw std::runtime_error("Unable to compile kernel");
	const auto fun = fun0.value();
	if (auto c = getConstantDeclarationName(s); !c.empty())
	{
		CUdeviceptr ptr = fun.constant(c);
		fillConstantMemory(s, ptr, histogramStream_);
	}

	//launch kernel
	int gridSize = 1;
	int blockSize = fun.bestBlockSize();
	const void* args[] = { &histogramOutDevice };
	auto result = cuLaunchKernel(
		fun.fun(), gridSize, 1, 1, blockSize, 1, 1,
		0, histogramStream_, const_cast<void**>(args), NULL);
	if (result != CUDA_SUCCESS) {
		printError(result, kernelName);
		return nullptr;
	}

	//copy results back
	CUMAT_SAFE_CALL(cudaMemcpyAsync(
		&histogramHost_, histogramOutDevice, sizeof(kernel::VolumeHistogram),
		cudaMemcpyDeviceToHost, histogramStream_));

	//record event for completion notifiation
	CUMAT_SAFE_CALL(cudaEventRecord(histogramCompletionEvent_, histogramStream_));
	histogramExtractionRunning_ = true;
	histogramCurrentKey_ = key;

	return nullptr; //not done yet
}

void renderer::VolumeInterpolationGrid::loadVolumeDialog()
{
	std::cout << "Open file dialog" << std::endl;

	// open file dialog
	std::string volumeDirectory = volumeFullFilename_.parent_path().string();
	auto results = pfd::open_file(
		"Load volume",
		volumeDirectory,
		{ "Volumes", "*.dat *.xyz *.cvol *.json" },
		false
	).result();
	if (results.empty())
		return;
	std::string fileNameStr = results[0];

	ImGui::MarkIniSettingsDirty();
	ImGui::SaveIniSettingsToDisk(GImGui->IO.IniFilename);
	loadVolumeFromPath(fileNameStr);
}

void renderer::VolumeInterpolationGrid::loadVolumeFromPath(const std::string& filename)
{
	std::cout << "Load " << filename << std::endl;
	auto fileNamePath = std::filesystem::path(filename);
	if (fileNamePath.extension() == ".json")
	{
	    //load ensemble
		loadEnsemble(filename, nullptr);
		newVolumeLoaded_ = true;
		return;
	}

	//load the file
	worker_.wait(); //wait for current task
	std::shared_ptr<float> progress = std::make_shared<float>(0);
	auto guiTask = [progress]()
	{
		std::cout << "Progress " << *progress.get() << std::endl;
		if (ImGui::BeginPopupModal("Load Volume", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
		{
			ImGui::ProgressBar(*progress.get(), ImVec2(200, 0));
			ImGui::EndPopup();
		}
	};
	this->backgroundGui_ = guiTask;
	ImGui::OpenPopup("Load Volume");
	auto loaderTask = [filename, progress, this](BackgroundWorker* worker)
	{
		loadVolume(filename, progress.get());

		//set it in the GUI and close popup
		this->backgroundGui_ = {};
		ImGui::CloseCurrentPopup();
		newVolumeLoaded_ = true;
	};
	//start background task
	worker_.launch(loaderTask);
}

void renderer::VolumeInterpolationGrid::loadEnsemble(const std::string& filename, float* progress)
{
	VolumeEnsembleFactory_ptr f;
	try
	{
		f = std::make_shared<VolumeEnsembleFactory>(filename);
		volumeFullFilename_ = filename;
		setSource(f);
	} catch (const std::exception& ex)
	{
		std::cerr << "Unable to load ensemble factory: " << ex.what();
		ensembleFactory_ = nullptr;
		volume_ = nullptr;
		source_ = VolumeSource::EMPTY;
	}
}

void renderer::VolumeInterpolationGrid::loadVolume(const std::string& filename, float* progress)
{
	ensembleFactory_ = nullptr;
	Volume_ptr volume = loadVolumeImpl(filename, progress);
	if (volume != nullptr) {
		setSource(volume, 0);
		volumeFullFilename_ = filename;
		std::cout << "Loaded" << std::endl;
	}
}

renderer::Volume_ptr renderer::VolumeInterpolationGrid::loadVolumeImpl(const std::string& filename, float* progress)
{
	static std::unordered_map<std::string, std::weak_ptr<Volume>> CACHE;
	if (auto it = CACHE.find(filename); it != CACHE.end())
	{
		auto ptr = it->second.lock();
		if (ptr) return ptr;
	}

	auto fileNamePath = std::filesystem::path(filename);
	//callbacks
	renderer::VolumeProgressCallback_t progressCallback = [progress](float v)
	{
		if (progress) *progress = v * 0.99f;
	};
	renderer::VolumeLoggingCallback_t logging = [](const std::string& msg)
	{
		std::cout << msg << std::endl;
	};
	int errorCode = 1;
	renderer::VolumeErrorCallback_t error = [&errorCode](const std::string& msg, int code)
	{
		errorCode = code;
		std::cerr << msg << std::endl;
		throw std::runtime_error(msg.c_str());
	};

	//load it locally
	try {
		std::shared_ptr<renderer::Volume> volume;
		if (fileNamePath.extension() == ".dat") {
			volume = Volume::loadVolumeFromRaw(filename, progressCallback, logging, error);
		}
		else if (fileNamePath.extension() == ".xyz") {
			volume = Volume::loadVolumeFromXYZ(filename, progressCallback, logging, error);
		}
		else if (fileNamePath.extension() == ".cvol")
			volume = std::make_shared<Volume>(filename, progressCallback, logging, error);
		else {
			std::cerr << "Unrecognized extension: " << fileNamePath.extension() << std::endl;
		}
		if (volume != nullptr) {
			CACHE[filename] = volume;
			return volume;
		}
	}
	catch (const std::exception& ex)
	{
		std::cerr << "Unable to load volume: " << ex.what() << std::endl;
	}
	return {};
}

void renderer::VolumeInterpolationGrid::load(const nlohmann::json& json, const ILoadingContext* fetcher)
{
	source_ = magic_enum::enum_cast<VolumeSource>(json.value("source", "")).
		value_or(VolumeSource::EMPTY);
	interpolation_ = magic_enum::enum_cast<VolumeInterpolation>(json.value("interpolation", "")).
		value_or(VolumeInterpolation::TRILINEAR);

	if (source_ == VolumeSource::VOLUME)
	{
		volumeFullFilename_ = std::filesystem::path();
		volume_ = nullptr;
		auto filename = std::filesystem::path(json.value("volumePath", ""));
		if (filename.is_relative())
			filename = absolute(fetcher->getRootPath() / filename);
		filename = filename.lexically_normal();
		if (std::filesystem::exists(filename))
		{
			std::cout << "Load volume " << filename << std::endl;
			if (filename.extension() == ".json")
			{
				//load ensemble
				loadEnsemble(filename.string(), nullptr);
			}
			else {
				loadVolume(filename.string(), nullptr);
				mipmapLevel_ = json.value("mipmapLevel", 0);
				if (volume_ && mipmapLevel_ != 0)
				{
					setSource(volume_, mipmapLevel_);
				}
			}
		}
		if (!volume_)
		{
			std::cerr << "Unable to load volume, is the file valid? " << filename << std::endl;
			source_ = VolumeSource::EMPTY;
		}
		else
		{
			volumeFullFilename_ = filename;
		}
		selectedDensityFeatureIndex_ = json.value("selectedDensityFeatureIndex", 0);
		selectedVelocityFeatureIndex_ = json.value("selectedVelocityFeatureIndex", 0);
		selectedColorFeatureIndex_ = json.value("selectedColorFeatureIndex", 0);
	}
}

void renderer::VolumeInterpolationGrid::save(nlohmann::json& json, const ISavingContext* context) const
{
	json["source"] = magic_enum::enum_name(source_);
	json["interpolation"] = magic_enum::enum_name(interpolation_);
	if (source_ == VolumeSource::VOLUME)
	{
#if 1
		//save relative path
		auto path = std::filesystem::relative(volumeFullFilename_, context->getRootPath());
		//test if the conversion was successful (i.e. same drive)
		auto path2 = absolute(context->getRootPath() / path);
		if (canonical(absolute(volumeFullFilename_)) != canonical(path2))
		{
			std::cout << "Volume file does not reside on the same filesystem as the settings file\n"
				<< " -> save as absolute path" << std::endl;
			path = volumeFullFilename_;
		}

#else
		//save absolute path
		auto path = std::filesystem::absolute(volumeFullFilename_);
#endif
		auto pathStr = path.string();
		std::replace(pathStr.begin(), pathStr.end(), '\\', '/'); //make platform independent
		json["volumePath"] = pathStr;
		json["mipmapLevel"] = mipmapLevel_;
		json["selectedDensityFeatureIndex"] = selectedDensityFeatureIndex_;
		json["selectedVelocityFeatureIndex"] = selectedVelocityFeatureIndex_;
		json["selectedColorFeatureIndex"] = selectedColorFeatureIndex_;
	}
}

void renderer::VolumeInterpolationGrid::registerPybindModule(pybind11::module& m)
{
	IVolumeInterpolation::registerPybindModule(m);

	//guard double registration
	static bool registered = false;
	if (registered) return;
	registered = true;
	
	namespace py = pybind11;
	py::class_<VolumeInterpolationGrid, IVolumeInterpolation, std::shared_ptr<VolumeInterpolationGrid>> c(m, "VolumeInterpolationGrid");
	py::enum_<VolumeSource>(c, "VolumeSource")
		.value("Volume", VolumeSource::VOLUME)
		.value("TorchTensor", VolumeSource::TORCH_TENSOR)
		.value("Empty", VolumeSource::EMPTY)
		.export_values();
	py::enum_<VolumeInterpolation>(c, "VolumeInterpolation")
		.value("NearestNeighbor", VolumeInterpolation::NEAREST_NEIGHBOR)
		.value("Trilinear", VolumeInterpolation::TRILINEAR)
		.value("Tricubic", VolumeInterpolation::TRICUBIC)
		.export_values();
	c.def(py::init<>())
		.def("source", &VolumeInterpolationGrid::source)
		.def("interpolation", &VolumeInterpolationGrid::interpolation)
		.def("setInterpolation", &VolumeInterpolationGrid::setInterpolation)
		.def("minDensity", &VolumeInterpolationGrid::minDensity)
		.def("maxDensity", &VolumeInterpolationGrid::maxDensity)
		.def("volume", &VolumeInterpolationGrid::volume)
		.def("mipmap_level", &VolumeInterpolationGrid::mipmapLevel)
		.def("tensor", &VolumeInterpolationGrid::tensor)
		.def("setSource", static_cast<void (VolumeInterpolationGrid::*)(Volume_ptr, int)>(&VolumeInterpolationGrid::setSource))
		.def("setSource", static_cast<void (VolumeInterpolationGrid::*)(const torch::Tensor&)>(&VolumeInterpolationGrid::setSource))
		.def("setSource", static_cast<void (VolumeInterpolationGrid::*)(VolumeEnsembleFactory_ptr)>(&VolumeInterpolationGrid::setSource))
		.def("setEnsembleAndTime", &VolumeInterpolationGrid::setEnsembleAndTime,
			py::arg("ensemble"), py::arg("time"), py::arg("mipmap")=0)
	    .def_property("grid_resolution_new_behavior", 
			&VolumeInterpolationGrid::isGridResolutionNewBehavior,
			&VolumeInterpolationGrid::setGridResolutionNewBehavior,
			py::doc(R"(
Switches between old and now behavior for the conversion from normalized world coords to object coords.
Old: multiply position with (resolution-1), introduces slight offset when using mipmaps
New: multiply position with (resolution), no directed offset, global shrinking
Default: old behavior
)"))
		;
	//TODO: specify feature to use
}

renderer::IKernelModule::GlobalSettings::VolumeOutput renderer::VolumeInterpolationGrid::outputType() const
{
	//TODO: for now, always the first density feature is used.
	//The Python binding does not yet expose a way to specify the feature
	return GlobalSettings::VolumeOutput::Density;
}

std::optional<int> renderer::VolumeInterpolationGrid::getBatches(const GlobalSettings& s) const
{
	if (source_ == VolumeSource::TORCH_TENSOR)
		return tensor_.value.size(0);
	return {};
}

std::string renderer::VolumeInterpolationGrid::getDefines(const GlobalSettings& s) const
{
	if (s.volumeShouldProvideCurvature && !s.volumeShouldProvideNormals)
	{
		throw std::runtime_error("Curvature estimation requested, but this requires also normals, which are deactivated");
	}

	std::stringstream ss;
	if (s.volumeShouldProvideNormals)
		ss << "#define VOLUME_INTERPOLATION_GRID__REQUIRES_NORMAL\n";

	if (source_ == VolumeSource::TORCH_TENSOR) {
		ss << "#define VOLUME_INTERPOLATION_GRID__USE_TENSOR\n";
		if (tensor_.value.dtype() == c10::kDouble)
			ss << "#define VOLUME_INTERPOLATION_GRID__TENSOR_TYPE double\n";
		else
			ss << "#define VOLUME_INTERPOLATION_GRID__TENSOR_TYPE float\n";
	}

	switch (interpolation_)
	{
	case VolumeInterpolation::NEAREST_NEIGHBOR:
		ss << "#define VOLUME_INTERPOLATION_GRID__INTERPOLATION 0\n";
		break;
	case VolumeInterpolation::TRILINEAR:
		ss << "#define VOLUME_INTERPOLATION_GRID__INTERPOLATION 1\n";
		break;
	case VolumeInterpolation::TRICUBIC:
		ss << "#define VOLUME_INTERPOLATION_GRID__INTERPOLATION 2\n";
		break;
	}

	if (s.interpolationInObjectSpace)
		ss << "#define VOLUME_INTERPOLATION_GRID__OBJECT_SPACE\n";

	if (source_ == VolumeSource::VOLUME) {
		switch (s.volumeOutput)
		{
		case GlobalSettings::Density:
		    {
			const auto feature = availableDensityFeatures_[selectedDensityFeatureIndex_];
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE " << Feature2DensityTextureType[int(feature.mapping)] << "\n";
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR " << Feature2DensityTextureChannel[int(feature.mapping)] << "\n";
		    if (feature.mapping==Feature2Density::DensityCurvature)
		    {
				ss << "#define VOLUME_INTERPOLATION_GRID__CURVATURE_FROM_GRID\n";
		    }
			break;
		    }
		case GlobalSettings::Velocity:
		    {
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE real4\n";
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR getVelocity\n";
			break;
		    }
		case GlobalSettings::Color:
			{
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE real4\n";
			ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR getColor\n";
			break;
			}
		}
	} else
	{
		//fall-back
		if (s.volumeOutput != GlobalSettings::Density)
			throw std::runtime_error("Unsupported operation: interpolation from PyTorch tensors is only supported for density output at the moment.");
		ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_TYPE float\n";
		ss << "#define VOLUME_INTERPOLATION_GRID__TEXTURE_EXTRACTOR getDirect\n";
	}

	if (gridResolutionNewBehavior_)
	{
		ss << "#define VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR 0\n";
	} else
	{
		ss << "#define VOLUME_INTERPOLATION_GRID__GRID_RESOLUTION_OLD_BEHAVIOR 1\n";
	}

	return ss.str();
}

std::vector<std::string> renderer::VolumeInterpolationGrid::getIncludeFileNames(const GlobalSettings& s) const
{
	return {
		"renderer_volume_grid.cuh"
	};
}

std::string renderer::VolumeInterpolationGrid::getConstantDeclarationName(const GlobalSettings& s) const
{
	return "volumeInterpolationGridParameters";
}

std::string renderer::VolumeInterpolationGrid::getPerThreadType(const GlobalSettings& s) const
{
	return "::kernel::VolumeInterpolationGrid";
}

void renderer::VolumeInterpolationGrid::fillConstantMemory(
	const GlobalSettings& s, CUdeviceptr ptr, CUstream stream)
{
	switch (source_)
	{
	case VolumeSource::EMPTY:
		throw std::runtime_error("No volume specified, can't render!");
	case VolumeSource::TORCH_TENSOR:
	{
		const auto& tensor = tensor_.value;
		if (!tensor.defined())
			throw std::runtime_error("No tensor specified (tensor.defined()==false)");
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "VolumeInterpolationGrid", [&]()
		{
			using real3 = typename ::kernel::scalar_traits<scalar_t>::real3;
			::kernel::Tensor4Read<scalar_t> tensorAcc = accessor<::kernel::Tensor4Read<scalar_t>>(tensor);
			struct Parameters
			{
				::kernel::Tensor4Read<scalar_t> tensor;
				int3 resolutionMinusOne; //resolution-1
				real3 boxMin;
				real3 boxSize;
				real3 normalStep;
				real3 normalScale;
			} p;
			p.tensor = tensorAcc;
			int3 objectResolution = make_int3(tensor.size(1), tensor.size(2), tensor.size(3));
			p.resolutionMinusOne = objectResolution - make_int3(1);
			p.boxMin = kernel::cast3<scalar_t>(boxMin());
			p.boxSize = kernel::cast3<scalar_t>(boxSize());
			double3 voxelSize = boxSize() / make_double3(
				objectResolution + make_int3(isGridResolutionNewBehavior() ? 0 : -1));
			p.normalScale = kernel::cast3<scalar_t>(0.5 / voxelSize); //central differences
			double3 normalStep = make_double3(1);//make_double3(1.0) / make_double3(objectResolution());
			p.normalStep = kernel::cast3<scalar_t>(normalStep);
			CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
		});
	}break;
	case VolumeSource::VOLUME:
	{
		if (!volume_)
			throw std::runtime_error("No volume specified, can't render!");
	    int featureIndex;
		switch (s.volumeOutput)
		{
		case GlobalSettings::Density: {
			if (availableDensityFeatures_.empty())
				throw std::runtime_error("Selected volume does not contain any scalar features. Can't render!");
			const auto mapping = availableDensityFeatures_[selectedDensityFeatureIndex_];
			featureIndex = mapping.featureIndex;
            }
			break;
		case GlobalSettings::Velocity: {
			if (availableVelocityFeatures_.empty())
				throw std::runtime_error("Selected volume does not contain any velocity features. Can't render!");
			featureIndex = availableVelocityFeatures_[selectedVelocityFeatureIndex_].featureIndex;
            }
			break;
		case GlobalSettings::Color: {
			if (availableColorFeatures_.empty())
				throw std::runtime_error("Selected volume does not contain any color features. Can't render!");
			featureIndex = availableColorFeatures_[selectedColorFeatureIndex_].featureIndex;
            }
			break;
		default:
			throw std::runtime_error("unknown output mode");
		}
		auto level = volume_->getFeature(featureIndex)->getLevel(mipmapLevel_);
		level->copyCpuToGpu();
		RENDERER_DISPATCH_FLOATING_TYPES(s.scalarType, "VolumeInterpolationGrid", [&]()
		{
			using real3 = typename ::kernel::scalar_traits<scalar_t>::real3;
			struct Parameters
			{
				cudaTextureObject_t tex;
				int3 resolutionMinusOne; //resolution-1
				real3 boxMin;
				real3 boxSize;
				real3 normalStep;
				real3 normalScale;
			} p;
			if (interpolation_ == VolumeInterpolation::NEAREST_NEIGHBOR)
				p.tex = level->dataTexGpuNearest();
			else
				p.tex = level->dataTexGpuLinear();
			p.resolutionMinusOne = objectResolution() - make_int3(1);
			p.boxMin = kernel::cast3<scalar_t>(boxMin());
			p.boxSize = kernel::cast3<scalar_t>(boxSize());
			double3 voxelSize = boxSize() / make_double3(
				objectResolution() + make_int3(isGridResolutionNewBehavior() ? 0 : -1));
			p.normalScale = kernel::cast3<scalar_t>(0.5 / voxelSize); //central differences
			double3 normalStep = make_double3(1);//make_double3(1.0) / make_double3(objectResolution());
			p.normalStep = kernel::cast3<scalar_t>(normalStep);
			CU_SAFE_CALL(cuMemcpyHtoDAsync(ptr, &p, sizeof(Parameters), stream));
		});
	}break;
	}
}


