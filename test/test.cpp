#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest.h"

#include <silarray.h>
#include <cstring>

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (std::strcmp(argv[i], "--cpu") == 0) {
      sil::use_cpu();
    } else if (std::strcmp(argv[i], "--gpu") == 0) {
      sil::use_mps();
    }
  }

  doctest::Context context;
  context.applyCommandLine(argc, argv);
  return context.run();
}
