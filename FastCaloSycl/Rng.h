//
// GeoLoad.hpp
//

#ifndef __FASTCALOSYCL_RNG_HPP__
#define __FASTCALOSYCL_RNG_HPP__

// mkl/sycl includes
#include <CL/sycl.hpp>

#include "mkl.h"
#include "mkl_sycl.hpp"

// example parameters defines
#define SEED 777
#define N 1000
#define N_PRINT 10

class Rng {
 public:
  Rng() : m_dev() {
    
    //
    // Initialization
    //

    // Catch asynchronous exceptions
    auto exception_handler = [](cl::sycl::exception_list exceptions) {
      for (std::exception_ptr const& e : exceptions) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception const& e) {
          std::cout << "Caught asynchronous SYCL exception during generation:\n"
                    << e.what() << std::endl;
        }
      }
    };

    cl::sycl::queue main_queue(m_dev, exception_handler);

    cl::sycl::context context = main_queue.get_context();

    // set range for RNG
    float a = -10.0;
    float b = 10.0;

    mkl::rng::philox4x32x10 engine(main_queue, SEED);

    mkl::rng::uniform<float> distribution(a, b);

    // prepare array for random numbers
    cl::sycl::usm_allocator<float, cl::sycl::usm::alloc::shared, 64> allocator(
        context, m_dev);
    std::vector<
        float, cl::sycl::usm_allocator<float, cl::sycl::usm::alloc::shared, 64>>
        r(N, allocator);

    //
    // Perform generation
    //

    try {
      auto event_out = mkl::rng::generate(distribution, engine, N, r.data());
    } catch (cl::sycl::exception const& e) {
      std::cout << "\t\tSYCL exception during generation\n"
                << e.what() << std::endl
                << "OpenCl status: " << e.get_cl_code() << std::endl;
      return;
    }

    main_queue.wait_and_throw();
  }

  ~Rng() {}

 private:
  cl::sycl::device m_dev;
};
#endif  // __FASTCALOSYCL_RNG_HPP__