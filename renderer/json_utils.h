#pragma once

#include <iostream>
#include <vector_functions.h>
#include "json.hpp"

namespace nlohmann {
	template <>
	struct adl_serializer<float3> {
		static void to_json(json& j, const float3& v) {
			j = json::array({ v.x, v.y, v.z });
		}

		static void from_json(const json& j, float3& v) {
			if (j.is_array() && j.size() == 3)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a float3" << std::endl;
		}
	};

	template <>
	struct adl_serializer<double3> {
		static void to_json(json& j, const double3& v) {
			j = json::array({ v.x, v.y, v.z });
		}

		static void from_json(const json& j, double3& v) {
			if (j.is_array() && j.size() == 3)
			{
				v.x = j.at(0).get<double>();
				v.y = j.at(1).get<double>();
				v.z = j.at(2).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a double3" << std::endl;
		}
	};

	template <>
	struct adl_serializer<float4> {
		static void to_json(json& j, const float4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, float4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
				v.w = j.at(3).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a float4" << std::endl;
		}
	};

	template <>
	struct adl_serializer<double4> {
		static void to_json(json& j, const double4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, double4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<double>();
				v.y = j.at(1).get<double>();
				v.z = j.at(2).get<double>();
				v.w = j.at(3).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a double4" << std::endl;
		}
	};

	template <>
	struct adl_serializer<glm::vec3> {
		static void to_json(json& j, const glm::vec3& v) {
			j = json::array({ v.x, v.y, v.z });
		}

		static void from_json(const json& j, glm::vec3& v) {
			if (j.is_array() && j.size() == 3)
			{
				v.x = j.at(0).get<float>();
				v.y = j.at(1).get<float>();
				v.z = j.at(2).get<float>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a glm::vec3" << std::endl;
		}
	};

	template <>
	struct adl_serializer<glm::vec4> {
		static void to_json(json& j, const glm::vec4& v) {
			j = json::array({ v.x, v.y, v.z, v.w });
		}

		static void from_json(const json& j, glm::vec4& v) {
			if (j.is_array() && j.size() == 4)
			{
				v.x = j.at(0).get<double>();
				v.y = j.at(1).get<double>();
				v.z = j.at(2).get<double>();
				v.w = j.at(3).get<double>();
			}
			else
				std::cerr << "Unable to deserialize " << j << " into a double4" << std::endl;
		}
	};
}
