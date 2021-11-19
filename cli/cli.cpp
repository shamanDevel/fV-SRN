#include "CLI11.hpp"

#include <string>
#include <indicators/cursor_control.hpp>
#include <indicators/progress_bar.hpp>
#include <volume.h>

std::shared_ptr<renderer::Volume> loadVolume(const std::string& inputFile, std::optional<int> ensemble)
{
    using namespace indicators;

    indicators::ProgressBar bar{
      option::BarWidth{50},
      option::Start{"\r ["},
      option::Fill{"."},
      option::Lead{" "},
      option::Remainder{" "},
      option::End{"]"},
      option::PrefixText{"Load Volume"},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true}
    };

    renderer::VolumeProgressCallback_t progressCallback = [&bar](float v)
    {
        bar.set_progress(static_cast<int>(100 * v));
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
    };

    std::filesystem::path fileNamePath = inputFile;
    std::shared_ptr<renderer::Volume> volume;
    if (fileNamePath.extension() == ".dat")
        volume = renderer::Volume::loadVolumeFromRaw(inputFile, progressCallback, logging, error, ensemble);
    else if (fileNamePath.extension() == ".xyz")
        volume = renderer::Volume::loadVolumeFromXYZ(inputFile, progressCallback, logging, error);
    else if (fileNamePath.extension() == ".cvol")
        volume = std::make_unique<renderer::Volume>(inputFile, progressCallback, logging, error);
    else {
        std::cerr << "Unrecognized extension: " << fileNamePath.extension() << std::endl;
    }

    return volume;
}

void saveVolume(const renderer::Volume* volume, const std::string& outputFile, int compression)
{
    using namespace indicators;

    indicators::ProgressBar bar{
      option::BarWidth{50},
      option::Start{"\r ["},
      option::Fill{"."},
      option::Lead{" "},
      option::Remainder{" "},
      option::End{"]"},
      option::PrefixText{"Save Volume"},
      option::ShowElapsedTime{true},
      option::ShowRemainingTime{true}
    };

    renderer::VolumeProgressCallback_t progressCallback = [&bar](float v)
    {
        bar.set_progress(static_cast<int>(100 * v));
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
    };

    volume->save(outputFile, progressCallback, logging, error, compression);
}

void convert(const std::string& inputFile, const std::string& outputFile, std::optional<int> ensemble, int compression)
{
    std::cout << "Convert \"" << inputFile << "\" to \"" <<
        outputFile << "\" using compression level " << compression << std::endl;

	if (outputFile.size()<5 || outputFile.substr(outputFile.size()-5)!=".cvol")
	{
        std::cerr << "Output filename does not end with \".cvol\", will not be loadable in the GUI" << std::endl;
	}

    indicators::show_console_cursor(false);
    const auto volume = loadVolume(inputFile, ensemble);
    if (volume)
        saveVolume(volume.get(), outputFile, compression);
    indicators::show_console_cursor(true);
}

int main(int argc, char** argv) {
    CLI::App app{ "CLI for volume rendering" };

    //subcommand for volume conversion
    CLI::App* convertCom = app.add_subcommand("convert", "Volume conversion");
    std::string convertInputFile, convertOutputFile;
    int convertCompression = 0;
    int convertEnsemble = -1;
    convertCom->add_option("-c,--compression", convertCompression, "compression rate, between 0 (no compression) and 9 (max compression)", true)
        ->check(CLI::Range(0, renderer::Volume::MAX_COMPRESSION));
    auto convertEnsembleOp = convertCom->add_option("-e,--ensemble", convertEnsemble, "which ensemble member should be loaded", true);
    convertCom->add_option("input", convertInputFile, "input file name")
        ->check(CLI::ExistingFile);
    convertCom->add_option("output", convertOutputFile, "output file name, saved as .cvol.");

    CLI11_PARSE(app, argc, argv);

    if (convertCom->parsed())
    {
        convert(convertInputFile, convertOutputFile,
            convertEnsemble>=0 ? std::optional<int>(convertEnsemble) : std::optional<int>{},
            convertCompression);
    }
    else
    {
        std::cerr << "No command specified, terminating" << std::endl;
    }
	
    return 0;
}