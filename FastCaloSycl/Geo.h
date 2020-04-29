//
// DeviceGeo.h
//

#ifndef FASTCALOSYCL_GEO_H_
#define FASTCALOSYCL_GEO_H_

#include <CL/sycl.hpp>
#include <map>

#include "CaloDetDescrElement.h"
#include "GeoRegion.h"

typedef std::map<Identifier, const CaloDetDescrElement*> cellmap_t;

// Stores information about a given sampling; sampling index and the "size",
// i.e. number of regions in the sampling.
struct SampleIndex {
  unsigned int size;
  int index;
};

// Geometry information to reside on device.
struct DeviceGeo {
  CaloDetDescrElement* cells;
  GeoRegion* regions;
  unsigned long num_cells;
  unsigned int num_regions;
  int sample_max;
  SampleIndex* sample_index;
};

// Geometry class.
// Stores calorimeter geometry information -- e.g. regions, cells, etc. -- that
// is read in from a CaloGeometryFromFile object. The data structure is designed
// to be compatible for a SYCL device, and can therefore be loaded to an
// accelerator.
// Host and device memory are allocated, and clients can access both the host
// and device allocated memory through accessor functions.
// The memory is freed upon destruction.
class Geo {
 public:
  Geo();
  ~Geo() { cl::sycl::free(&cells_device_, ctx_); }

  static struct DeviceGeo* device_geo_;

  void set_num_cells(unsigned long num_cells) { num_cells_ = num_cells; }
  void set_num_regions(unsigned int num_regions) { num_regions_ = num_regions; }
  void set_cell_map(cellmap_t* cell_map) { cell_map_ = cell_map; }
  void set_host_regions(GeoRegion* regions) { regions_ = regions; }
  void set_device_regions(GeoRegion* regions) { regions_device_ = regions; }
  void set_host_cells(CaloDetDescrElement* cells) { cells_ = cells; }
  void set_device_cells(CaloDetDescrElement* cells) { cells_device_ = cells; }
  void set_max_sample(int sample) { max_sample_ = sample; }
  void set_sample_index(SampleIndex* index) { sample_index_ = index; }
  const CaloDetDescrElement* index_to_cell(unsigned long index) {
    return (*cell_map_)[cell_id_[index]];
  }

  // Copy geometry to device using USM. Returns true if device memory is
  // allocated and data is copied, and false otherwise.
  bool LoadDeviceGeo();

 private:
  bool AllocMemCells(cl::sycl::device* dev);

  unsigned long num_cells_;            // Number of cells
  unsigned int num_regions_;           // Number of regions
  cellmap_t* cell_map_;                // From Geometry class(?)
  GeoRegion* regions_;                 // Regions on host
  GeoRegion* regions_device_;          // Regions on device
  CaloDetDescrElement* cells_;         // Cells on host
  CaloDetDescrElement* cells_device_;  // Cells on device
  Identifier* cell_id_;                // Cell ID to Identifier lookup table
  unsigned int max_sample_;            // Max number of samples
  SampleIndex* sample_index_;          // Needed?
  cl::sycl::context ctx_;  // SYCL device context; needed for freeing memory
};
#endif  // FASTCALOSYCL_GEO_H_