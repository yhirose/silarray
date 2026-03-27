#pragma once

#include <objc/objc.h>
#include <objc/runtime.h>
#include <objc/message.h>
#include <CoreFoundation/CoreFoundation.h>

namespace sil {
namespace objc {

struct mtl_size {
  unsigned long width, height, depth;
};

inline SEL sel(const char* name) {
  return sel_registerName(name);
}

// Pre-cached selectors for hot-path ObjC calls
namespace sel_ {
  inline SEL alloc() { static auto s = sel("alloc"); return s; }
  inline SEL release() { static auto s = sel("release"); return s; }
  inline SEL contents() { static auto s = sel("contents"); return s; }
  inline SEL length() { static auto s = sel("length"); return s; }
  inline SEL commandBuffer() { static auto s = sel("commandBuffer"); return s; }
  inline SEL computeCommandEncoder() { static auto s = sel("computeCommandEncoder"); return s; }
  inline SEL endEncoding() { static auto s = sel("endEncoding"); return s; }
  inline SEL commit() { static auto s = sel("commit"); return s; }
  inline SEL waitUntilCompleted() { static auto s = sel("waitUntilCompleted"); return s; }
  inline SEL setPSO() { static auto s = sel("setComputePipelineState:"); return s; }
  inline SEL setBuffer() { static auto s = sel("setBuffer:offset:atIndex:"); return s; }
  inline SEL setBytes() { static auto s = sel("setBytes:length:atIndex:"); return s; }
  inline SEL dispatch() { static auto s = sel("dispatchThreads:threadsPerThreadgroup:"); return s; }
  inline SEL dispatchTG() { static auto s = sel("dispatchThreadgroups:threadsPerThreadgroup:"); return s; }
  inline SEL memoryBarrier() { static auto s = sel("memoryBarrierWithScope:"); return s; }
  inline SEL initBuffer() { static auto s = sel("initWithBuffer:offset:descriptor:"); return s; }
  inline SEL newBuffer() { static auto s = sel("newBufferWithLength:options:"); return s; }
  inline SEL mpsDesc() { static auto s = sel("matrixDescriptorWithRows:columns:rowBytes:dataType:"); return s; }
  inline SEL mpsMatmulInit() { static auto s = sel("initWithDevice:transposeLeft:transposeRight:resultRows:resultColumns:interiorColumns:alpha:beta:"); return s; }
  inline SEL mpsEncode() { static auto s = sel("encodeToCommandBuffer:leftMatrix:rightMatrix:resultMatrix:"); return s; }
}  // namespace sel_

// Basic message sends — selector is cached on first call per call site
inline void* send(void* obj, const char* sel_name) {
  return reinterpret_cast<void*(*)(void*, SEL)>(objc_msgSend)(
      obj, sel(sel_name));
}

inline void* send(void* obj, SEL s) {
  return reinterpret_cast<void*(*)(void*, SEL)>(objc_msgSend)(obj, s);
}

inline void* send(void* obj, SEL s, void* a, size_t b, size_t c) {
  return reinterpret_cast<void*(*)(void*, SEL, void*, size_t, size_t)>(
      objc_msgSend)(obj, s, a, b, c);
}

inline void* send(void* obj, const char* sel_name, void* a) {
  return reinterpret_cast<void*(*)(void*, SEL, void*)>(objc_msgSend)(
      obj, sel(sel_name), a);
}

inline void* send(void* obj, const char* sel_name, void* a, size_t b, size_t c) {
  return reinterpret_cast<void*(*)(void*, SEL, void*, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b, c);
}

inline void* send(void* obj, const char* sel_name, const void* a, size_t b, size_t c) {
  return reinterpret_cast<void*(*)(void*, SEL, const void*, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b, c);
}

inline void* send(void* obj, const char* sel_name, size_t a, size_t b) {
  return reinterpret_cast<void*(*)(void*, SEL, size_t, size_t)>(
      objc_msgSend)(obj, sel(sel_name), a, b);
}

inline size_t send_uint(void* obj, const char* sel_name) {
  return reinterpret_cast<size_t(*)(void*, SEL)>(objc_msgSend)(
      obj, sel(sel_name));
}

// Hot-path sends with pre-cached selectors
inline void send_set_pso(void* enc, void* pso) {
  reinterpret_cast<void(*)(void*, SEL, void*)>(objc_msgSend)(
      enc, sel_::setPSO(), pso);
}
inline void send_set_buffer(void* enc, void* buf, size_t off, size_t idx) {
  reinterpret_cast<void(*)(void*, SEL, void*, size_t, size_t)>(objc_msgSend)(
      enc, sel_::setBuffer(), buf, off, idx);
}
inline void send_set_bytes(void* enc, const void* data, size_t len, size_t idx) {
  reinterpret_cast<void(*)(void*, SEL, const void*, size_t, size_t)>(objc_msgSend)(
      enc, sel_::setBytes(), data, len, idx);
}

inline void send_dispatch(void* obj, mtl_size grid, mtl_size threads) {
  reinterpret_cast<void(*)(void*, SEL, mtl_size, mtl_size)>(objc_msgSend)(
      obj, sel_::dispatch(), grid, threads);
}

// MTLBarrierScopeBuffers = 1
inline void send_memory_barrier(void* enc, unsigned long scope = 1) {
  reinterpret_cast<void(*)(void*, SEL, unsigned long)>(objc_msgSend)(
      enc, sel_::memoryBarrier(), scope);
}

// MPS-specific sends
inline void* send_mps_desc(void* cls, size_t rows, size_t cols,
                           size_t row_bytes, unsigned long dtype) {
  return reinterpret_cast<void*(*)(void*, SEL, size_t, size_t, size_t,
                                   unsigned long)>(objc_msgSend)(
      cls, sel_::mpsDesc(), rows, cols, row_bytes, dtype);
}

inline void* send_mps_matmul_init(void* obj, void* device,
                                   bool tl, bool tr,
                                   size_t rows, size_t cols, size_t inner,
                                   double alpha, double beta) {
  return reinterpret_cast<void*(*)(void*, SEL, void*, bool, bool,
                                   size_t, size_t, size_t,
                                   double, double)>(objc_msgSend)(
      obj, sel_::mpsMatmulInit(), device, tl, tr, rows, cols, inner, alpha, beta);
}

inline void send_mps_encode(void* matmul, void* cb, void* left, void* right,
                            void* result) {
  reinterpret_cast<void(*)(void*, SEL, void*, void*, void*, void*)>(
      objc_msgSend)(matmul, sel_::mpsEncode(), cb, left, right, result);
}

// NSString / CFString creation
inline void* cfstr(const char* s) {
  return (void*)CFStringCreateWithCString(nullptr, s, kCFStringEncodingUTF8);
}

inline void cfrelease(void* ref) {
  CFRelease((CFTypeRef)ref);
}

inline void* cls(const char* name) {
  return (void*)objc_getClass(name);
}

inline void release(void* obj) {
  reinterpret_cast<void(*)(void*, SEL)>(objc_msgSend)(obj, sel_::release());
}

// NSArray<NSNumber*> from an array of ints (used for MPSShape / permutation).
// Returned array is autoreleased — caller should retain if it outlives the pool.
inline void* ns_numbers(const int* dims, size_t n) {
  static auto NSNumber_cls = cls("NSNumber");
  static auto NSArray_cls = cls("NSArray");
  static auto numWithInt = sel("numberWithInt:");
  static auto arrayWith = sel("arrayWithObjects:count:");
  void* nums[8];  // conv2d shapes/permutations are always 4 dims; 8 is ample
  for (size_t i = 0; i < n; i++) {
    nums[i] = reinterpret_cast<void*(*)(void*, SEL, int)>(objc_msgSend)(
        NSNumber_cls, numWithInt, dims[i]);
  }
  return reinterpret_cast<void*(*)(void*, SEL, void**, size_t)>(objc_msgSend)(
      NSArray_cls, arrayWith, nums, n);
}

inline void* retain(void* obj) {
  static auto s = sel("retain");
  return reinterpret_cast<void*(*)(void*, SEL)>(objc_msgSend)(obj, s);
}

}  // namespace objc
};  // namespace sil
