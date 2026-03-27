#pragma once

#include <unified_memory.h>

namespace sil {

enum class Device {
  MPS,
  CPU,
};

inline Device device_ = Device::MPS;
inline bool gpu_pending_ = false;

inline void use_cpu() { device_ = Device::CPU; }
inline void use_mps() { device_ = Device::MPS; }

};  // namespace sil
