#include <catch.hpp>

#include <tinyformat.h>
namespace fmt = tinyformat;

TEST_CASE("Ensemble-Formatting", "[modules]")
{
	const int ensemble = 5;
	const int time = 7;
	const auto format = [&](const std::string& f)
	{
		return fmt::format(f.c_str(), ensemble, time);
	};

	const std::string format1 = "files/ensemble%1$03d/time%2$03d.dat";
	REQUIRE("files/ensemble005/time007.dat" == format(format1));

	const std::string format2 = "files/time%2$03d.dat";
	REQUIRE("files/time007.dat" == format(format2));

	const std::string format3 = "files/ensemble%1$03d.dat";
	REQUIRE("files/ensemble005.dat" == format(format3));
}