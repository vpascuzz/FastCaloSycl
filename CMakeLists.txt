# Copyright (C) 2002-2020 CERN for the benefit of the ATLAS collaboration

cmake_minimum_required(VERSION 3.10)

set( CMAKE_C_COMPILER "clang" )
set( CMAKE_CXX_COMPILER "clang++" )

# Special make options require for CUDA backend
if(ENABLE_PI_CUDA)
  message(STATUS "Using SYCL backend: Plugin Interface CUDA")
  add_definitions( -DUSE_PI_CUDA )
  set( CMAKE_CXX_FLAGS "-g -O2 -fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version" )
else()
  set( CMAKE_CXX_FLAGS "-g -O2 -fsycl -std=c++17 -Wno-unknown-cuda-version" )
endif()

# Sources
set(FastCaloSycl_Srcs
  src/Geo.cc
  src/GeoRegion.cc
)

# Define and build the library
message(STATUS "Creating library target '${FastCaloSycl_LIB}'")
add_library(${FastCaloSycl_LIB} SHARED ${FastCaloSycl_Srcs})
target_link_libraries( ${FastCaloSycl_LIB}
                        ${AthenaStandalone_LIB}
                        ${FastCaloSimAnalyzer_LIB}
                        ${FastCaloSimCommon_LIB} )
target_include_directories( ${FastCaloSycl_LIB}
                            PRIVATE
                            FastCaloSycl
                            ${${FastCaloSimAnalyzer_LIB}_Includes}
                            ${${AthenaStandalone_LIB}_Includes}
                            ${${FastCaloSimCommon_LIB}_Includes} )

# Install library
install(TARGETS ${FastCaloSycl_LIB}
  DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Test(s) in the package
if(ENABLE_SYCL_TESTS)
  if(ENABLE_PI_CUDA)
    message(STATUS "Using SYCL backend: Plugin Interface CUDA")
    set( CMAKE_CXX_FLAGS "-g -O2 -fsycl -std=c++17 -fsycl-targets=nvptx64-nvidia-cuda-sycldevice -Wno-unknown-cuda-version" )
  else()
    set( CMAKE_CXX_FLAGS "-g -O2 -fsycl -std=c++17 -Wno-unknown-cuda-version" )
  endif()
  
  # Set default directory for test installs
  set (CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_TEST_OUTPUT_DIRECTORY})

  # Test executables
  if(!ENABLE_PI_CUDA) # MKL is not compatible with CUDA
    add_executable( Rng_test test/Rng_test.cc )
    target_link_libraries( Rng_test
                          PRIVATE
                          mkl_sycl mkl_intel_ilp64 mkl_tbb_thread mkl_core
                          tbb sycl OpenCL pthread m dl )
  endif()
  add_executable( GeoRegion_test
                  test/GeoRegion_test.cc
                  src/GeoRegion.cc )
  target_link_libraries( GeoRegion_test
                        ${AthenaStandalone_LIB}
                        ${FastCaloSimAnalyzer_LIB}
                        ${FastCaloSimCommon_LIB} )
  target_include_directories( GeoRegion_test
                              PRIVATE
                              FastCaloSycl
                              ${${FastCaloSimAnalyzer_LIB}_Includes}
                              ${${AthenaStandalone_LIB}_Includes}
                              ${${FastCaloSimCommon_LIB}_Includes} )
  # Tests that use SYCL
  add_executable( Sycl_test
                  test/Sycl_test.cc
                  src/GeoRegion.cc
                  src/Geo.cc )
  target_link_libraries( Sycl_test
                        ${AthenaStandalone_LIB}
                        ${FastCaloSimAnalyzer_LIB}
                        ${FastCaloSimCommon_LIB} )
  target_include_directories( Sycl_test
                              PRIVATE
                              FastCaloSycl
                              ${${AthenaStandalone_LIB}_Includes}
                              ${${FastCaloSimAnalyzer_LIB}_Includes}
                              ${${FastCaloSimCommon_LIB}_Includes} )
endif()