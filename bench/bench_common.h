#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <mach/mach.h>

struct BenchEntry {
  std::string name;
  double seconds;
};

using BenchGroup = std::pair<std::string, std::vector<BenchEntry>>;

// Ensure GPU is in a high-performance power state before measurement.
// Apple GPUs clock down after ~ms of idle, inflating subsequent results.
inline void gpu_warmup() {
  auto a = sil::random({512, 512});
  auto b = sil::random({512, 512});
  sil::synchronize();
  for (int i = 0; i < 10; i++) { auto c = a.dot(b); sil::synchronize(); }
}

// Auto-init: called once per process via static init.
// Warms up GPU and ggml metal backend before any benchmarks run.
namespace bench_init_ {
  inline bool warmup_done = [] {
    gpu_warmup();
    return true;
  }();
}

// Duration-based throughput measurement with outlier rejection.
// Runs multiple short trials (~0.5s each), takes median to reject outliers.
// Warmup stabilizes GPU power state before measurement.
inline double measure(size_t /*unused*/, auto fn,
                      int warmup_ms = 30, int trials = 3) {
  using clk = std::chrono::high_resolution_clock;
  using sec = std::chrono::duration<double>;

  // Warmup: stabilize GPU power state
  auto warm_end = clk::now() + std::chrono::milliseconds(warmup_ms);
  while (clk::now() < warm_end) fn();

  // Calibrate: determine iteration count for ~50ms per trial
  int cal_count = 0;
  auto cal_start = clk::now();
  auto cal_end = cal_start + std::chrono::milliseconds(50);
  while (clk::now() < cal_end) { fn(); cal_count++; }
  int trial_iters = std::max(cal_count, 1);

  // Measure: take median to reject outliers
  std::vector<double> times(trials);
  for (int t = 0; t < trials; t++) {
    auto t0 = clk::now();
    for (int i = 0; i < trial_iters; i++) fn();
    auto t1 = clk::now();
    times[t] = sec(t1 - t0).count() / trial_iters;
  }
  std::sort(times.begin(), times.end());
  return times[trials / 2];
}

inline double gflops_gemm(size_t M, size_t N, size_t K, double seconds) {
  return 2.0 * M * N * K / seconds / 1e9;
}

inline size_t peak_rss_bytes() {
  mach_task_basic_info_data_t info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
  return info.resident_size;
}

inline void print_group(const BenchGroup& group, int name_width = 0) {
  constexpr int bar_width = 50;

  auto& [title, entries] = group;

  if (name_width == 0) {
    name_width = 5;
    for (auto& e : entries)
      name_width = std::max(name_width, static_cast<int>(e.name.size()));
  }

  std::printf("  %s\n", title.c_str());

  double fastest =
      std::ranges::min_element(entries, {}, &BenchEntry::seconds)->seconds;

  for (auto& e : entries) {
    double ratio = e.seconds / fastest;
    int bars = std::max(1, static_cast<int>(std::round(bar_width / ratio)));
    bool is_fastest = (e.seconds == fastest);

    std::string bar;
    for (int i = 0; i < bars; ++i) bar += "\xe2\x96\x85";
    for (int i = bars; i < bar_width; ++i) bar += ' ';

    auto time_str = e.seconds < 1e-3
                        ? std::format("{:>9.2f} us", e.seconds * 1e6)
                        : std::format("{:>9.2f} ms", e.seconds * 1e3);

    if (is_fastest)
      std::printf("    %-*s \x1b[36m%s\x1b[0m %s %6.1fx\n", name_width,
                  e.name.c_str(), bar.c_str(), time_str.c_str(), ratio);
    else
      std::printf("    %-*s %s %s %6.1fx\n", name_width, e.name.c_str(),
                  bar.c_str(), time_str.c_str(), ratio);
  }
  std::puts("");
  std::fflush(stdout);
}

inline void print_section(const char* name) {
  std::printf("\n─ %s ─────────────────────────────────────\n\n", name);
  std::fflush(stdout);
}

inline void print_bar_graphs(const std::vector<BenchGroup>& groups) {
  int name_width = 5;
  for (auto& [title, entries] : groups)
    for (auto& e : entries)
      name_width = std::max(name_width, static_cast<int>(e.name.size()));

  auto section_name = [](const std::string& title) {
    auto pos = title.find(" (");
    return pos != std::string::npos ? title.substr(0, pos) : title;
  };

  std::string current_section;
  for (auto& group : groups) {
    auto sec = section_name(group.first);
    if (sec != current_section) {
      current_section = sec;
      print_section(sec.c_str());
    }
    print_group(group, name_width);
  }
}

inline void print_csv(const std::vector<BenchGroup>& groups) {
  std::printf("benchmark,library,best_us\n");
  for (auto& [title, entries] : groups) {
    for (auto& e : entries) {
      std::printf("%s,%s,%.2f\n", title.c_str(), e.name.c_str(),
                  e.seconds * 1e6);
    }
  }
}

inline void print_table(const std::vector<BenchGroup>& groups,
                        const char* title = nullptr,
                        const char* description = nullptr) {
  if (title) std::printf("### %s\n\n", title);
  if (description) std::printf("%s\n\n", description);
  // Collect all library names in order of first appearance
  std::vector<std::string> libs;
  for (auto& [title, entries] : groups)
    for (auto& e : entries)
      if (std::ranges::find(libs, e.name) == libs.end())
        libs.push_back(e.name);

  // Format a time value compactly
  auto fmt_time = [](double s) -> std::string {
    if (s < 1e-3) return std::format("{:.0f} us", s * 1e6);
    if (s < 1.0)  return std::format("{:.2f} ms", s * 1e3);
    return std::format("{:.2f} s", s);
  };

  // Find max benchmark name width
  int name_w = 9; // "benchmark"
  for (auto& [title, entries] : groups)
    name_w = std::max(name_w, static_cast<int>(title.size()));

  int col_w = 10; // minimum column width

  // Header
  std::printf("| %-*s |", name_w, "benchmark");
  for (auto& lib : libs) std::printf(" %-*s |", col_w, lib.c_str());
  std::puts("");

  // Separator
  std::printf("|");
  for (int i = 0; i < name_w + 2; i++) std::printf("-");
  std::printf("|");
  for (size_t i = 0; i < libs.size(); i++) {
    for (int j = 0; j < col_w + 2; j++) std::printf("-");
    std::printf("|");
  }
  std::puts("");

  // Rows
  for (auto& [title, entries] : groups) {
    double fastest = std::ranges::min_element(entries, {}, &BenchEntry::seconds)->seconds;

    std::printf("| %-*s |", name_w, title.c_str());

    for (auto& lib : libs) {
      auto it = std::ranges::find_if(entries, [&](auto& e) { return e.name == lib; });
      if (it != entries.end()) {
        auto time = fmt_time(it->seconds);
        double ratio = it->seconds / fastest;
        auto cell = (ratio <= 1.05) ? std::format("**{}**", time)
                                    : std::format("{} ({:.1f}x)", time, ratio);
        std::printf(" %-*s |", col_w, cell.c_str());
      } else {
        std::printf(" %-*s |", col_w, "-");
      }
    }
    std::puts("");
  }
  std::puts("");
  std::fflush(stdout);
}

enum class OutputMode { bar, csv, table };

inline OutputMode parse_output_mode(int argc, const char** argv) {
  for (int i = 1; i < argc; i++) {
    if (std::string("--csv") == argv[i]) return OutputMode::csv;
    if (std::string("--table") == argv[i]) return OutputMode::table;
  }
  return OutputMode::bar;
}

// Keep old API working
inline bool has_csv_flag(int argc, const char** argv) {
  return parse_output_mode(argc, argv) == OutputMode::csv;
}
