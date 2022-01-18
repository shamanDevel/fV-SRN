#include "visualizer.h"

#include <filesystem>
#include <fstream>
#include <iomanip>
//For the timestamp of the screenshots
#ifdef _MSC_VER
#define NOMINMAX
#include <Windows.h>
#else
#include <time.h>
#ifndef MAX_PATH
#define MAX_PATH 260
#endif
#endif

#include <cuMat/src/Errors.h>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <tinyformat.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <imgui.h>
#include <IconsFontAwesome5.h>
#include <imgui_extension.h>
#include <imgui_internal.h>

#include <json.hpp>
#include <lodepng.h>
#include <portable-file-dialogs.h>
#include <magic_enum.hpp>

#include "module_registry.h"
#include "utils.h"

const char* Visualizer::BackgroundColorNames[_BACKGROUND_COLOR_COUNT_] = {
	"Black", "White"
};
const ImVec4 Visualizer::BackgroundColors[_BACKGROUND_COLOR_COUNT_] = {
	ImVec4(0,0,0,0),
	ImVec4(1,1,1,1)
};

Visualizer::Visualizer(GLFWwindow* window)
	: window_(window)
{
	// Add .ini handle for ImGuiWindow type
	ImGuiSettingsHandler ini_handler;
	ini_handler.TypeName = "Visualizer";
	ini_handler.TypeHash = ImHashStr("Visualizer");
	static const auto replaceWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == ' ') cpy[i] = '%'; //'%' is not allowed in path names
		return cpy;
	};
	static const auto insertWhitespace = [](const std::string& s) -> std::string
	{
		std::string cpy = s;
		for (int i = 0; i < cpy.size(); ++i)
			if (cpy[i] == '%') cpy[i] = ' '; //'%' is not allowed in path names
		return cpy;
	};
	auto settingsReadOpen = [](ImGuiContext*, ImGuiSettingsHandler* handler, const char* name) -> void*
	{
		return handler->UserData;
	};
	auto settingsReadLine = [](ImGuiContext*, ImGuiSettingsHandler* handler, void* entry, const char* line)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		char path[MAX_PATH];
		int intValue = 0;
		memset(path, 0, sizeof(char)*MAX_PATH);
		std::cout << "reading \"" << line << "\"" << std::endl;
		if (sscanf(line, "SettingsDir=%s", path) == 1)
			vis->settingsDirectory_ = insertWhitespace(std::string(path));
		if (sscanf(line, "SettingsToLoad=%d", &intValue) == 1)
			vis->settingsToLoad_ = intValue;
	};
	auto settingsWriteAll = [](ImGuiContext* ctx, ImGuiSettingsHandler* handler, ImGuiTextBuffer* buf)
	{
		Visualizer* vis = reinterpret_cast<Visualizer*>(handler->UserData);
		buf->reserve(200);
		buf->appendf("[%s][Settings]\n", handler->TypeName);
		std::string settingsDirectory = replaceWhitespace(vis->settingsDirectory_);
		std::cout << "Write settings:" << std::endl;
		buf->appendf("SettingsDir=%s\n", settingsDirectory.c_str());
		buf->appendf("SettingsToLoad=%d\n", vis->settingsToLoad_);
		buf->appendf("\n");
	};
	ini_handler.UserData = this;
	ini_handler.ReadOpenFn = settingsReadOpen;
	ini_handler.ReadLineFn = settingsReadLine;
	ini_handler.WriteAllFn = settingsWriteAll;
	GImGui->SettingsHandlers.push_back(ini_handler);

	//initialize renderer
	if (!renderer::KernelLoader::Instance().initCuda())
	{
		std::cerr << "CUDA not found!" << std::endl;
		exit(-1);
	}
	//renderer::KernelLoader::Instance().setDebugMode(true);
	renderer::ModuleRegistry::Instance().populateModules();
	const auto& evaluators =
		renderer::ModuleRegistry::Instance().getModulesForTag(renderer::IImageEvaluator::Tag());
	selectedImageEvaluator_ = std::dynamic_pointer_cast<renderer::IImageEvaluator>(evaluators[0].first);
}

Visualizer::~Visualizer()
{
	releaseResources();
}

void Visualizer::releaseResources()
{
	if (screenTextureCuda_)
	{
		CUMAT_SAFE_CALL(cudaGraphicsUnregisterResource(screenTextureCuda_));
		screenTextureCuda_ = nullptr;
	}
	if (screenTextureGL_)
	{
		glDeleteTextures(1, &screenTextureGL_);
		screenTextureGL_ = 0;
	}
	if (screenTextureCudaBuffer_)
	{
		CUMAT_SAFE_CALL(cudaFree(screenTextureCudaBuffer_));
		screenTextureCudaBuffer_ = nullptr;
	}
}

void Visualizer::settingsSave()
{
	// save file dialog
	auto fileNameStr = pfd::save_file(
		"Save settings",
		settingsDirectory_,
		{ "Json file", "*.json" },
		true
	).result();
	if (fileNameStr.empty())
		return;

	auto fileNamePath = std::filesystem::path(fileNameStr);
	fileNamePath = fileNamePath.replace_extension(".json");
	auto rootPath = absolute(fileNamePath.parent_path());
	std::cout << "Save settings to " << fileNamePath << std::endl;
	settingsDirectory_ = fileNamePath.string();

	// Build json
	nlohmann::json settings;
	settings["version"] = 2;
	settings["root"] = selectedImageEvaluator_->getName();
	renderer::ModuleRegistry::Instance().saveAll(settings, rootPath);
	settings["backgroundColor"] = backgroundColor_;

	//save json to file
	std::ofstream o(fileNamePath);
	o << std::setw(4) << settings << std::endl;
	footerString_ = std::string("Settings saved to ") + fileNamePath.string();
	footerTimer_ = 2.0f;
}

namespace
{
	std::string getDir(const std::string& path)
	{
		if (path.empty())
			return path;
		std::filesystem::path p(path);
		if (std::filesystem::is_directory(p))
			return path;
		return p.parent_path().string();
	}
}

void Visualizer::settingsLoad()
{
	// load file dialog
	auto results = pfd::open_file(
        "Load settings",
        getDir(settingsDirectory_),
        { "Json file", "*.json" },
        false
    ).result();
	if (results.empty())
		return;

	auto fileNameStr = results[0];
	auto fileNamePath = std::filesystem::path(fileNameStr);
	auto rootPath = absolute(fileNamePath.parent_path());
	std::cout << "Load settings from " << fileNamePath << std::endl;
	settingsDirectory_ = rootPath.string();

	//load json
	std::ifstream i(fileNamePath);
	nlohmann::json settings;
	try
	{
		i >> settings;
	} catch (const nlohmann::json::exception& ex)
	{
		pfd::message("Unable to parse Json", std::string(ex.what()),
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}
	i.close();
	int version = settings.contains("version")
		? settings.at("version").get<int>()
		: 0;
	if (version != 2)
	{
		pfd::message("Illegal Json", "The loaded json does not contain settings in the correct format",
			pfd::choice::ok, pfd::icon::error).result();
		return;
	}
	backgroundColor_ = settings.value("backgroundColor", 0);

	//Ask which part should be loaded
	renderer::ModuleRegistry::Instance().loadAll(settings, rootPath);
	if (auto m = renderer::ModuleRegistry::Instance().getModule(
		renderer::IImageEvaluator::Tag(), settings.value("root", "")))
	{
		selectedImageEvaluator_ = std::dynamic_pointer_cast<renderer::IImageEvaluator>(m);
	}
	triggerRedraw(RedrawRenderer);
}

static void HelpMarker(const char* desc)
{
	//ImGui::TextDisabled(ICON_FA_QUESTION);
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted(desc);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}
void Visualizer::specifyUI()
{
	uiMenuBar();

	ImGui::PushItemWidth(ImGui::GetFontSize() * -8);

	bool changed = false;
	const auto& evaluators =
		renderer::ModuleRegistry::Instance().getModulesForTag(renderer::IImageEvaluator::Tag());
	if (evaluators.size() > 1) {
		for (int i = 0; i < evaluators.size(); ++i) {
			const auto& name = evaluators[i].first->getName();
			if (ImGui::RadioButton(name.c_str(), evaluators[i].first == selectedImageEvaluator_)) {
				selectedImageEvaluator_ = std::dynamic_pointer_cast<renderer::IImageEvaluator>(evaluators[i].first);
				changed = true;
			}
			if (i < evaluators.size() - 1) ImGui::SameLine();
		}
	}

	renderer::ModuleRegistry::mapTree(selectedImageEvaluator_, [&](renderer::IModule_ptr m)
		{
			if (m->updateUI(uiStorage_))
				changed = true;
		});
	
	if (selectedImageEvaluator_->drawUI(uiStorage_))
		changed = true;

	ImGui::SliderInt("Background##Main", &backgroundColor_, 0, _BACKGROUND_COLOR_COUNT_ - 1, BackgroundColorNames[backgroundColor_]);

	if (changed)
		triggerRedraw(RedrawRenderer);
	
	ImGui::PopItemWidth();

	if (backgroundGui_)
		backgroundGui_();

	uiFooterOverlay();
	uiFPSOverlay();
}

void Visualizer::uiMenuBar()
{
	ImGui::BeginMenuBar();
	ImGui::Text("Hotkeys");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
		ImGui::TextUnformatted("'P': Screenshot");
		ImGui::TextUnformatted("'L': Lock foveated center");
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
	if (ImGui::SmallButton("Save##Settings"))
		settingsSave();
	if (ImGui::SmallButton("Load##Settings"))
		settingsLoad();
	if (ImGui::SmallButton("Reload Kernels##Settings")) {
		renderer::KernelLoader::Instance().reloadCudaKernels();
		triggerRedraw(RedrawRenderer);
	}
	ImGui::EndMenuBar();
	//hotkeys
	if (ImGui::IsKeyPressed(GLFW_KEY_P, false))
	{
		screenshot();
	}
}

void Visualizer::uiFooterOverlay()
{
	if (footerTimer_ <= 0) return;

	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x / 2, io.DisplaySize.y - 10);
	ImVec2 window_pos_pivot = ImVec2(0.5f, 1.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	//ImGui::PushStyleVar(ImGuiStyleVar_Alpha, alpha);
	ImGui::Begin("Example: Simple overlay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::TextUnformatted(footerString_.c_str());
	ImGui::End();
	//ImGui::PopStyleVar(ImGuiStyleVar_Alpha);

	footerTimer_ -= io.DeltaTime;
}

void Visualizer::uiFPSOverlay()
{
	ImGuiIO& io = ImGui::GetIO();
	ImVec2 window_pos = ImVec2(io.DisplaySize.x - 5, 5);
	ImVec2 window_pos_pivot = ImVec2(1.0f, 0.0f);
	ImGui::SetNextWindowPos(window_pos, ImGuiCond_Always, window_pos_pivot);
	ImGui::SetNextWindowBgAlpha(0.5f);
	ImGui::Begin("FPSDisplay", nullptr, ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav);
	ImGui::Text("FPS: %.1f", io.Framerate);
	renderer::ModuleRegistry::mapTree(selectedImageEvaluator_, [&](renderer::IModule_ptr m)
		{
			m->drawExtraInfo(uiStorage_);
		});
	ImGui::End();
}

void Visualizer::render(int display_w, int display_h)
{
	resize(display_w, display_h);

	bool redraw = false;
	bool refine = false;
	if (redrawMode_ != RedrawNone)
	{
		redraw = true;
	}
	else if (selectedImageEvaluator_->isIterativeRefining())
	{
		redraw = true;
		refine = true;
	}

	if (redraw)
	{
		//recompute
		try
		{
			CUstream stream = renderer::IImageEvaluator::getDefaultStream();
			outputTensor_ = selectedImageEvaluator_->render(
				display_w, display_h, stream, refine, outputTensor_);
			selectedImageEvaluator_->copyOutputToTexture(outputTensor_, screenTextureCudaBuffer_,
				selectedImageEvaluator_->selectedChannel(), stream);
			auto result = cuStreamSynchronize(stream);
			if (result != CUDA_SUCCESS)
			{
				const char* pStr;
				cuGetErrorString(result, &pStr);
				throw std::runtime_error(tinyformat::format("Error on synchronization: %s", pStr));
			}
			copyBufferToOpenGL();
		} catch (const std::exception& ex)
		{
			static std::string lastError;
			const std::string newError(ex.what());
			if (lastError != newError) {
				std::cerr << "Rendering error:\n" << newError << std::endl;
				lastError = newError;
			}
			setFooterMessage("Rendering error:\n" + newError);
		}
		redrawMode_ = RedrawNone;
	}

	drawer_.drawQuad(screenTextureGL_);
}


void Visualizer::copyBufferToOpenGL()
{
	CUMAT_SAFE_CALL(cudaGraphicsMapResources(1, &screenTextureCuda_, 0));
	cudaArray* texture_ptr;
	CUMAT_SAFE_CALL(cudaGraphicsSubResourceGetMappedArray(&texture_ptr, screenTextureCuda_, 0, 0));
	size_t size_tex_data = sizeof(GLubyte) * displayWidth_ * displayHeight_ * 4;
	CUMAT_SAFE_CALL(cudaMemcpyToArray(texture_ptr, 0, 0, screenTextureCudaBuffer_, size_tex_data, cudaMemcpyDeviceToDevice));
	CUMAT_SAFE_CALL(cudaGraphicsUnmapResources(1, &screenTextureCuda_, 0));
}

void Visualizer::resize(int display_w, int display_h)
{
	//make it a nice multiplication of everything
	//const int multiply = 4 * 3;
	//display_w = display_w / multiply * multiply;
	//display_h = display_h / multiply * multiply;

	if (display_w == displayWidth_ && display_h == displayHeight_)
		return;
	if (display_w == 0 || display_h == 0)
		return;
	releaseResources();
	displayWidth_ = display_w;
	displayHeight_ = display_h;

	//create texture
	glGenTextures(1, &screenTextureGL_);
	glBindTexture(GL_TEXTURE_2D, screenTextureGL_);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // clamp s coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // clamp t coordinate
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	glTexImage2D(
		GL_TEXTURE_2D, 0, GL_RGBA8,
		displayWidth_, displayHeight_, 0
		, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

	//register with cuda
	CUMAT_SAFE_CALL(cudaGraphicsGLRegisterImage(
		&screenTextureCuda_, screenTextureGL_,
		GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard));

	//create channel output buffer
	CUMAT_SAFE_CALL(cudaMalloc(&screenTextureCudaBuffer_, displayWidth_ * displayHeight_ * 4 * sizeof(GLubyte)));

	glBindTexture(GL_TEXTURE_2D, 0);

	triggerRedraw(RedrawRenderer);
	std::cout << "Visualizer::resize(): " << displayWidth_ << ", " << displayHeight_ << std::endl;
}

void Visualizer::triggerRedraw(RedrawMode mode)
{
	redrawMode_ = std::max(redrawMode_, mode);
}


void Visualizer::setFooterMessage(const std::string& msg)
{
	footerString_ = msg;
	footerTimer_ = 2.0f;
}

void Visualizer::screenshot()
{
	std::string folder = "screenshots";

	char time_str[128];
	time_t now = time(0);
	struct tm* tstruct_ptr;
#ifdef _MSC_VER
	struct tm tstruct;
	localtime_s(&tstruct, &now);
	tstruct_ptr = &tstruct;
#else
	tstruct_ptr = localtime(&now);
#endif
	strftime(time_str, sizeof(time_str), "%Y%m%d-%H%M%S", tstruct_ptr);

	char output_name[512];
	sprintf(output_name, "%s/screenshot_%s.png", folder.c_str(), time_str);

	std::cout << "Take screenshot: " << output_name << std::endl;
	std::filesystem::create_directory(folder);

	std::vector<GLubyte> textureCpu(4 * displayWidth_ * displayHeight_);
	CUMAT_SAFE_CALL(cudaMemcpy(&textureCpu[0], screenTextureCudaBuffer_, 4 * displayWidth_*displayHeight_, cudaMemcpyDeviceToHost));

	if (lodepng_encode32_file(output_name, textureCpu.data(), displayWidth_, displayHeight_) != 0)
	{
		std::cerr << "Unable to save image" << std::endl;
		setFooterMessage(std::string("Unable to save screenshot to ") + output_name);
	}
	else
	{
		setFooterMessage(std::string("Screenshot saved to ") + output_name);
	}
}
