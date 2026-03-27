#pragma once

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

// Memory-mapped MNIST data loader.
// Data files expected at: ../test/ relative to bench/ directory.

struct mnist_file {
  void* addr = MAP_FAILED;
  size_t size = 0;
  int fd = -1;

  bool open(const char* path) {
    fd = ::open(path, O_RDONLY);
    if (fd == -1) return false;
    struct stat sb;
    fstat(fd, &sb);
    size = sb.st_size;
    addr = ::mmap(nullptr, size, PROT_READ, MAP_PRIVATE, fd, 0);
    return addr != MAP_FAILED;
  }

  ~mnist_file() {
    if (addr != MAP_FAILED) munmap(addr, size);
    if (fd != -1) close(fd);
  }

  const char* data() const { return (const char*)addr; }
};

struct mnist_data {
  static constexpr size_t kPixels = 784;   // 28x28
  static constexpr size_t kClasses = 10;

  size_t count = 0;
  std::vector<float> images;   // [count, 784] normalized to [0,1]
  std::vector<uint8_t> labels; // [count]

  bool load(const char* image_path, const char* label_path) {
    mnist_file img_file, lbl_file;
    if (!img_file.open(image_path) || !lbl_file.open(label_path)) {
      std::fprintf(stderr, "MNIST: cannot open %s or %s\n", image_path, label_path);
      return false;
    }

    auto read32 = [](const char* p) -> uint32_t {
      return __builtin_bswap32(*reinterpret_cast<const uint32_t*>(p));
    };

    if (read32(lbl_file.data()) != 2049 || read32(img_file.data()) != 2051) {
      std::fprintf(stderr, "MNIST: invalid file format\n");
      return false;
    }

    count = read32(lbl_file.data() + 4);
    labels.resize(count);
    std::memcpy(labels.data(), lbl_file.data() + 8, count);

    auto* pixels = reinterpret_cast<const uint8_t*>(img_file.data() + 16);
    images.resize(count * kPixels);
    for (size_t i = 0; i < count * kPixels; i++)
      images[i] = pixels[i] / 255.0f;

    return true;
  }
};
