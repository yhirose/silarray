#pragma once

#include <objc.h>

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

extern "C" void* MTLCreateSystemDefaultDevice(void);

namespace sil {

//-----------------------------------------------------------------------------

struct storage {
  std::shared_ptr<void> buf;
  void *data = nullptr;
  void *mtl_buf = nullptr;
  size_t off = 0;
  size_t len = 0;

  static storage make(size_t bytes);
};

//-----------------------------------------------------------------------------

class buffer_pool {
 public:
  void* device;

  static buffer_pool& instance() {
    static auto* pool = new buffer_pool();
    return *pool;
  }

  void* acquire(size_t bytes) {
    auto it = free_buffers_.find(bytes);
    if (it != free_buffers_.end() && !it->second.empty()) {
      auto buf = it->second.back();
      it->second.pop_back();
      return buf;
    }
    // MTLResourceStorageModeShared = 0
    return objc::send(device, "newBufferWithLength:options:", bytes, 0ul);
  }

  void release(void* buf, size_t bytes) {
    free_buffers_[bytes].push_back(buf);
  }

 private:
  std::unordered_map<size_t, std::vector<void*>> free_buffers_;

  buffer_pool() {
    device = MTLCreateSystemDefaultDevice();
    if (!device) {
      throw std::runtime_error("Failed to create Metal device.");
    }
  }
};

//-----------------------------------------------------------------------------

inline storage storage::make(size_t bytes) {
  auto& pool = buffer_pool::instance();
  auto* buf = pool.acquire(bytes);

  storage s;
  s.buf = std::shared_ptr<void>(buf, [bytes](void* p) {
    buffer_pool::instance().release(p, bytes);
  });
  s.data = objc::send(buf, objc::sel_::contents());
  s.mtl_buf = buf;
  s.off = 0;
  s.len = 0;
  return s;
}

};  // namespace sil
