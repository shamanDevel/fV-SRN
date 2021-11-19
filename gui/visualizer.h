#pragma once

#include <cuda_runtime.h>
#include <GL/glew.h>
#include <sstream>

#include "quad_drawer.h"
#include "background_worker.h"
#include <iimage_evaluator.h>

struct GLFWwindow;

class Visualizer
{
public:
	Visualizer(GLFWwindow* window);
	~Visualizer();

	void specifyUI();

	void render(int display_w, int display_h);
	ImVec4 getBackgroundColor() const { return BackgroundColors[backgroundColor_]; }

private:

	enum RedrawMode
	{
		RedrawNone,
		RedrawRenderer,

		_RedrawModeCount_
	};
	RedrawMode redrawMode_ = RedrawNone;

	GLFWwindow* window_;

	//main entry point
	renderer::IModule::UIStorage_t uiStorage_;
	renderer::IImageEvaluator_ptr selectedImageEvaluator_;
	torch::Tensor outputTensor_;
	//display
	int displayWidth_ = 0;
	int displayHeight_ = 0;
	unsigned int screenTextureGL_ = 0;
	cudaGraphicsResource_t screenTextureCuda_ = nullptr;
	GLubyte* screenTextureCudaBuffer_ = nullptr;
	QuadDrawer drawer_;
	enum BackgroundColor
	{
	    BACKGROUND_BLACK,
		BACKGROUND_WHITE,
		_BACKGROUND_COLOR_COUNT_
	};
	static const char* BackgroundColorNames[_BACKGROUND_COLOR_COUNT_];
	static const ImVec4 BackgroundColors[_BACKGROUND_COLOR_COUNT_];
	int backgroundColor_ = 0;

	//background computation
	BackgroundWorker worker_;
	std::function<void()> backgroundGui_;

	//screenshot
	std::string footerString_;
	float footerTimer_ = 0;

	//settings
	std::string settingsDirectory_;
	enum SettingsToLoad
	{
		CAMERA = 1,
		COMPUTATION_MODE = 2,
		TF_EDITOR = 4,
		RENDERER = 8,
		SHADING = 16,
		_ALL_SETTINGS_ = CAMERA | COMPUTATION_MODE | TF_EDITOR | RENDERER | SHADING
	};
	int settingsToLoad_ = _ALL_SETTINGS_;

private:
	void releaseResources();
	
	void settingsSave();
	void settingsLoad();

	void uiMenuBar();
	void uiFooterOverlay();
	void uiFPSOverlay();
	
	void copyBufferToOpenGL();
	void resize(int display_w, int display_h);
	void triggerRedraw(RedrawMode mode);

	void setFooterMessage(const std::string& msg);
	void screenshot();
};

