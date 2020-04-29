//
// Geo.cc
//

#include "Geo.h"

#include <chrono>

#ifdef __SYCL_DEVICE_ONLY__
#define CONSTANT __attribute__((opencl_constant))
#else
#define CONSTANT
#endif
static const CONSTANT char kCellInfo[] =
    "\tdevice_cell :: id [%llx], hash_id[%d]\n";
static const CONSTANT char kRegionInfo[] =
    "\tdevice_cell :: cellid64 [%llu], cellid[%llx]\n";

#ifdef USE_PI_CUDA
class CUDASelector : public cl::sycl::device_selector {
 public:
  int operator()(const cl::sycl::device& Device) const override {
    using namespace cl::sycl::info;

    const std::string DeviceName = Device.get_info<device::name>();
    const std::string DeviceVendor = Device.get_info<device::vendor>();
    const std::string DeviceDriver =
        Device.get_info<cl::sycl::info::device::driver_version>();

    if (Device.is_gpu() && (DeviceVendor.find("NVIDIA") != std::string::npos) &&
        (DeviceDriver.find("CUDA") != std::string::npos)) {
      return 1;
    };
    return -1;
  }
};
#endif

// Geo
Geo::Geo()
    : num_cells_(0UL),
      num_regions_(0),
      cell_map_(nullptr),
      regions_(nullptr),
      regions_device_(nullptr),
      cells_device_(nullptr),
      cell_id_(nullptr),
      max_sample_(0U),
      sample_index_(nullptr) {}

bool Geo::AllocMemCells(cl::sycl::device* dev) {
  // Allocate host-side memory for cell array.
  cells_ =
      (CaloDetDescrElement*)malloc(num_cells_ * sizeof(CaloDetDescrElement));
  cell_id_ = (Identifier*)malloc(num_cells_ * sizeof(Identifier));

  if (!cells_ || !cell_id_) {
    std::cout << "Cannot allocate host-side memory!" << std::endl;
    return false;
  }

  // Allocate device-side memory for cell array.
  cells_device_ = (CaloDetDescrElement*)malloc_device(
      num_cells_ * sizeof(CaloDetDescrElement), (*dev), ctx_);
  if (!cells_device_) {
    std::cout << "Cannot allocate device-side memory!" << std::endl;
    return false;
  }

  // Memory allocated successfully.
  std::cout << "Host and device cell memory allocated..." << std::endl;
  std::cout << "\tnum_cells: " << num_cells_ << std::endl
            << "\tsize: "
            << (int)num_cells_ * sizeof(CaloDetDescrElement) / 1000 << " kb\n"
            << std::endl;
  return true;
}  // AllocMemCells

// LoadDeviceGeo
DeviceGeo* Geo::device_geo_;
bool Geo::LoadDeviceGeo() {
  if (!cell_map_ || num_cells_ == 0) {
    std::cout << "Empty geometry!" << std::endl;
    return false;
  }

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

// Initialize device, queue and context
#ifdef USE_PI_CUDA
  CUDASelector cuda_selector;
  cl::sycl::device dev;
  try {
    dev = cl::sycl::device(cuda_selector);
  } catch (...) {
  }
#else
  cl::sycl::default_selector dev_selector;
  cl::sycl::device dev(dev_selector);
#endif
  cl::sycl::queue queue(dev, exception_handler);
  ctx_ = queue.get_context();
  // Name of the device to run on
  std::string dev_name =
      queue.get_device().get_info<cl::sycl::info::device::name>();
  std::cout << "Using device \"" << dev_name << "\"" << std::endl;

  // Ensure device can handle USM device allocations.
  if (!queue.get_device()
           .get_info<cl::sycl::info::device::usm_device_allocations>()) {
    std::cout << "ERROR :: device \"" << dev_name
              << "\" does not support usm_device_allocations!" << std::endl;
    return false;
  }

  //
  // CELLS
  //

  if (!AllocMemCells(&dev)) {
    std::cout << "ERROR :: Unable to allocate memory!" << std::endl;
    return false;
  }

  // Assign arrays for the host-side cells.
  int cell_index = 0;
  for (cellmap_t::iterator ic = cell_map_->begin(); ic != cell_map_->end();
       ++ic) {
    cells_[cell_index] = *(*ic).second;
    Identifier id = ((*ic).second)->identify();
    cell_id_[cell_index] = id;

    // Print host cell info
    if ((cell_index + 1) % 10000 == 0) {
      std::cout << "cell_id: " << std::hex << cells_[cell_index].identify()
                << std::dec << ", hash_id: " << cells_[cell_index].calo_hash()
                << std::endl;
    }
    cell_index++;
  }

  // Copy cell data to the device.
  std::cout << "Copying host cells to device... " << std::endl;
  // Start timer.
  auto geo_cpy_start = std::chrono::system_clock::now();
  auto ev_cpy_cells = queue.memcpy(cells_device_, &cells_[0],
                                   num_cells_ * sizeof(CaloDetDescrElement));
  ev_cpy_cells.wait_and_throw();
  // End timer.
  auto geo_cpy_end = std::chrono::system_clock::now();
  // Time to copy geometry host->device.
  auto geo_cpy_dur = std::chrono::duration<double>(geo_cpy_end - geo_cpy_start);
  std::cout << "\tCells copied in " << std::dec << geo_cpy_dur.count()
            << " ms.\n"
            << std::endl;

// Move this to unit test.
// CUDA does not support experimental::printf()
#ifndef USE_PI_CUDA
  std::cout << "Test device cells..." << std::endl;
  auto ev_cellinfo = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.parallel_for<class Dummy>(
        cl::sycl::range<1>(num_cells_),
        [=, dev_cells_local = this->cells_device_](cl::sycl::id<1> idx) {
          unsigned int id = (int)idx[0];
          if ((id + 1) % 10000 == 0) {
            long long cell_id = dev_cells_local[id].identify();
            unsigned long long hash = dev_cells_local[id].calo_hash();
            cl::sycl::intel::experimental::printf(kCellInfo, cell_id, hash);
          }
        });
  });
  ev_cellinfo.wait_and_throw();
#endif

  // Move this to a unit test.
  // Copy device cell data back to host.
  // std::cout << "Copying device cells to host... " << std::endl;
  // // Start timer.
  // auto geo_cpy_start2 = std::chrono::system_clock::now();
  // queue
  //     .memcpy(&host_cells_[0], device_cells_,
  //             num_cells_ * sizeof(CaloDetDescrElement))
  //     .wait_and_throw();
  // // End timer.
  // auto geo_cpy_end2 = std::chrono::system_clock::now();
  // // Time to copy geometry host->device.
  // auto geo_cpy_dur2 =
  //     std::chrono::duration<double>(geo_cpy_end2 - geo_cpy_start2);
  // std::cout << "\tCells copied in " << std::dec << geo_cpy_dur2.count()
  //           << " ms.\n"
  //           << std::endl;

  //
  // REGIONS
  //

  // Allocate device memory for each sampling's regions
  SampleIndex* device_si =
      (SampleIndex*)malloc_device(max_sample_ * sizeof(SampleIndex), dev, ctx_);

  // Copy sampling array to device
  auto ev_cpy_si = queue.memcpy(device_si, &sample_index_[0],
                                max_sample_ * sizeof(SampleIndex));
  ev_cpy_si.wait_and_throw();

  for (unsigned int iregion = 0; iregion < num_regions_; ++iregion) {
    int num_cells_eta = regions_[iregion].cell_grid_eta();
    int num_cells_phi = regions_[iregion].cell_grid_phi();

    // Allocate device memory for region cells
    long long* region_cells = (long long*)malloc_device(
        num_cells_eta * num_cells_phi * sizeof(long long), dev, ctx_);
    // Copy region cells to device
    auto ev_cpy_region_cells =
        queue.memcpy(region_cells, regions_[iregion].cell_grid(),
                     num_cells_eta * num_cells_phi * sizeof(long long));
    ev_cpy_region_cells.wait_and_throw();

    // Set cells pointer for region before copying to GPU so we know where the
    // cells are.
    regions_[iregion].set_cell_grid_device(region_cells);
    regions_[iregion].set_cells_device(cells_device_);
  }

  // Allocate region data memory and copy to device
  regions_device_ =
      (GeoRegion*)malloc_device(num_regions_ * sizeof(GeoRegion), dev, ctx_);
  auto ev_cpy_regions = queue.memcpy(regions_device_, &regions_[0],
                                     num_regions_ * sizeof(GeoRegion));
  ev_cpy_regions.wait_and_throw();

  // Device geometry
  DeviceGeo device_geo;
  device_geo.cells = cells_device_;
  device_geo.num_cells = num_cells_;
  device_geo.num_regions = num_regions_;
  device_geo.regions = regions_device_;
  device_geo.sample_max = max_sample_;
  device_geo.sample_index = device_si;

  // Copy device geometry to device, and set static member variable to this
  // pointer.
  DeviceGeo* device_geo_ptr =
      (DeviceGeo*)malloc_device(sizeof(DeviceGeo), dev, ctx_);
  auto ev_cpy_device_geo =
      queue.memcpy(device_geo_ptr, &device_geo, sizeof(DeviceGeo));
  device_geo_ = device_geo_ptr;

  // Test regions.
  // Move this to unit test.
  // CUDA does not support experimental::printf()
#ifndef USE_PI_CUDA
  std::cout << "Test device region..." << std::endl;
  auto ev_region_info = queue.submit([&](cl::sycl::handler& cgh) {
    cgh.single_task<class RegionTest>(
        [=, dev_regions_local = this->regions_device_]() {
          long long cellid64 =
              dev_regions_local[0].get_cell(-3.15858, 0.0545135, 0, 0);
          // int pos = dev_regions_local[0].raw_eta_pos_to_index(0.1);
          // unsigned long long cellid64(3179554531063103488);
          // Identifier cellid(cellid64);
          cl::sycl::intel::experimental::printf(kRegionInfo, cellid64, 2);
        });
  });
  ev_region_info.wait_and_throw();
#endif

  return true;
}  // LoadDeviceGeo