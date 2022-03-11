#pragma once

#include <variant>
#include <vector_types.h>
#include <torch/types.h>

BEGIN_RENDERER_NAMESPACE

enum class ParameterType
{
	Bool, Int, Int2, Int3, Int4,
	Double, Double2, Double3, Double4,
	Tensor
};
struct ParameterBase {};

/**
 * \brief A (differentiable) parameter of the renderer.
 * These parameters can be controlled by Python,
 * and contain those three fields:
 *  - value: the actual value (or multiple options if batched)
 *  - static constexpr bool supportsGradients
 *  - grad: the gradient tensor
 *  - forwardIndex: an integer tensor with the index into forward derivatives
 *
 * The values can usually be defined as a single scalar, e.g. 'double' or 'double4',
 * or as a torch::Tensor. This allows the parameter to be batched over the pixels/images.
 * The exact semantics depends on the use case.
 * The scalar parameters are stored as double-precision on the host.
 * Only during rendering they are cast to the actual scalar type (single or double precision).
 *
 * There does not exist a default implementation intentionally
 * to support only specific primitive parameter types.
 * 
 */
template<typename T>
struct Parameter;

#define PARAMETER_NON_DIFFERENTIABLE(T, PT)						\
template<>														\
struct Parameter<T> : ParameterBase								\
{																\
    using value_t = T;											\
	/* The value of this parameter: scalar or tensor for batching*/					\
	std::variant<T, torch::Tensor> value;										    \
	enum { supportsGradients = false };							\
	static constexpr ParameterType type = ParameterType::PT;	\
	Parameter() = default;															\
	Parameter(const T& v) : value(v) {}												\
	Parameter(const torch::Tensor& v) : value(v) {}									\
};
PARAMETER_NON_DIFFERENTIABLE(bool, Bool);
PARAMETER_NON_DIFFERENTIABLE(int, Int);
PARAMETER_NON_DIFFERENTIABLE(int2, Int2);
PARAMETER_NON_DIFFERENTIABLE(int3, Int3);
PARAMETER_NON_DIFFERENTIABLE(int4, Int4);
#undef PARAMETER_NON_DIFFERENTIABLE

#define PARAMETER_DIFFERENTIABLE(T, PT)												\
template<>																			\
struct Parameter<T> : ParameterBase												    \
{																					\
	/* The scalar type of this parameter */											\
	using value_t = T;															    \
	/* The value of this parameter: scalar or tensor for batching*/					\
	std::variant<T, torch::Tensor> value;										    \
	/* Gradients are supported */													\
	enum { supportsGradients = true };												\
	static constexpr ParameterType type = ParameterType::PT;						\
	/* If defined and of the same shape as 'value', will contain the gradients*/	\
	torch::Tensor grad;																\
	/* The indices for the forward method*/											\
	torch::Tensor forwardIndex;														\
	Parameter() = default;															\
	Parameter(const T& v) : value(v) {}												\
	Parameter(const torch::Tensor& v) : value(v) {}									\
};
PARAMETER_DIFFERENTIABLE(double , Double);
PARAMETER_DIFFERENTIABLE(double2, Double2);
PARAMETER_DIFFERENTIABLE(double3, Double3);
PARAMETER_DIFFERENTIABLE(double4, Double4);
#undef PARAMETER_DIFFERENTIABLE

template<>
struct Parameter<torch::Tensor> : ParameterBase
{
	using value_t = torch::Tensor;
	torch::Tensor value;
	enum { supportsGradients = true };
	static constexpr ParameterType type = ParameterType::Tensor;
	torch::Tensor grad;
	torch::Tensor forwardIndex;
};

/**
 * For use in the UI: enforces that the parameter
 * stores a scalar value (instead of the batched tensor)
 * and returns a pointer to the parameter for ImGUI.
 * If the parameter is stored as batched tensor,
 * replace it by 'alternative'.
 */
template<typename T>
T* enforceAndGetScalar(Parameter<T>& p, const T& alternative = T())
{
	static_assert(std::is_same_v<
		decltype(p.value),
		std::variant<T, torch::Tensor>>,
		"The parameter does not store its value as variant<T, Tensor>.");
	//already stored as scalar
	if (std::holds_alternative<T>(p.value))
		return &std::get<T>(p.value);
	//enforce scalar
	p.value = alternative;
	return &std::get<T>(p.value);
}

template<typename T>
const T* getScalarOrThrow(const Parameter<T>& p)
{
	static_assert(std::is_same_v<
		decltype(p.value),
		std::variant<T, torch::Tensor>>,
		"The parameter does not store its value as variant<T, Tensor>.");
	//already stored as scalar
	if (std::holds_alternative<T>(p.value))
		return &std::get<T>(p.value);
	//stored as tensor -> throw
	throw std::runtime_error("parameter stored as batched tensor, can't save it");
}
template<typename T>
T* getScalarOrThrow(Parameter<T>& p)
{
	static_assert(std::is_same_v<
		decltype(p.value),
		std::variant<T, torch::Tensor>>,
		"The parameter does not store its value as variant<T, Tensor>.");
	//already stored as scalar
	if (std::holds_alternative<T>(p.value))
		return &std::get<T>(p.value);
	//stored as tensor -> throw
	throw std::runtime_error("parameter stored as batched tensor, can't save it");
}


END_RENDERER_NAMESPACE
