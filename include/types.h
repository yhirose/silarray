#pragma once

#include <concepts>

namespace sil {

template <typename T>
concept value_type =
    std::same_as<T, float> || std::same_as<T, int> || std::same_as<T, bool>;

};  // namespace sil
