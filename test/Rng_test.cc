//
// Rng_test.cc
//

#include "../FastCaloSycl/Rng.h"

void test1() {
  std::cout << "*** Rng_test BEGIN" << std::endl;
  Rng* r = new Rng();
  if (r) {
    delete (r);
    r = nullptr;
  }
  std::cout << "*** Rng_test END" << std::endl;
}

int main() {
  test1();
  return 0;
}