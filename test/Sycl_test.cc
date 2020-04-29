//
// Sycl_test.cc
//

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <assert.h>

#include <CL/sycl.hpp>
#include <iostream>

#include "../FastCaloSycl/Geo.h"
#include "../FastCaloSycl/GeoRegion.h"
#include "FastCaloSimAnalyzer/CaloGeometryFromFile.h"
#include "TFCSSampleDiscovery.h"

CaloGeometryFromFile* get_calo_geo() {
  std::cout << "- Start get_calo_geo()" << std::endl;
  // Load calorimeter geometry from file
  CaloGeometryFromFile* geo = new CaloGeometryFromFile();
  assert(geo);

// Hard-code path for now
// TODO: Fix this!
#ifdef FCS_INPUT_PATH
  static const std::string geo_path_root(FCS_INPUT_PATH);
#else
  static const std::string geo_path_root =
      // "/home/vrpascuzzi/data/FastCaloSimInputs/CaloGeometry";
      "/bld2/data/FastCaloSimInputs/CaloGeometry";
#endif
  std::string geo_path_fcal1 =
      geo_path_root + "/FCal1-electrodes.sorted.HV.09Nov2007.dat";
  std::string geo_path_fcal2 =
      geo_path_root + "/FCal2-electrodes.sorted.HV.April2011.dat";
  std::string geo_path_fcal3 =
      geo_path_root + "/FCal3-electrodes.sorted.HV.09Nov2007.dat";
  geo->LoadGeometryFromFile(
      geo_path_root + "/Geometry-ATLAS-R2-2016-01-00-01.root",
      TFCSSampleDiscovery::geometryTree(),
      geo_path_root + "/cellId_vs_cellHashId_map.txt");
  geo->LoadFCalGeometryFromFiles(
      {geo_path_fcal1, geo_path_fcal2, geo_path_fcal3});

  std::cout << "- Done get_calo_geo()" << std::endl;
  return geo;
}

void copy_geo_region(CaloGeometryLookup* gl, GeoRegion* gr) {
  unsigned int num_eta = gl->cell_grid_eta();
  unsigned int num_phi = gl->cell_grid_phi();
  gr->set_xy_grid_adjust(gl->xy_grid_adjustment_factor());
  gr->set_index(gl->index());
  gr->set_cell_grid_eta(num_eta);
  gr->set_cell_grid_phi(num_phi);
  gr->set_min_eta(gl->mineta());
  gr->set_min_phi(gl->minphi());
  gr->set_max_eta(gl->maxeta());
  gr->set_max_phi(gl->maxphi());
  gr->set_deta(gl->deta());
  gr->set_dphi(gl->dphi());
  gr->set_min_eta_raw(gl->mineta_raw());
  gr->set_min_phi_raw(gl->minphi_raw());
  gr->set_max_eta_raw(gl->maxeta_raw());
  gr->set_max_phi_raw(gl->maxphi_raw());
  gr->set_eta_corr(gl->eta_correction());
  gr->set_phi_corr(gl->phi_correction());
  gr->set_min_eta_corr(gl->mineta_correction());
  gr->set_min_phi_corr(gl->minphi_correction());
  gr->set_max_eta_corr(gl->maxeta_correction());
  gr->set_max_phi_corr(gl->maxphi_correction());
  gr->set_deta_double(gl->deta_double());
  gr->set_dphi_double(gl->dphi_double());

  // Now load cells
  long long* cells = (long long*)malloc(sizeof(long long) * num_eta * num_phi);
  gr->set_cell_grid(cells);

  if (num_eta != (*(gl->cell_grid())).size()) {
    std::cout << "num_eta: " << num_eta
              << ", cell_grid: " << (*gl->cell_grid()).size() << std::endl;
  }

  // Loop over eta
  unsigned int ncells = 0;
  unsigned int empty_cells = 0;
  for (unsigned int ieta = 0; ieta < num_eta; ++ieta) {
    if (num_phi != (*(gl->cell_grid()))[ieta].size()) {
      std::cout << "num_phi: " << num_phi
                << ", cell_grid: " << (*gl->cell_grid())[ieta].size()
                << std::endl;
    }
    // Loop over phi
    for (unsigned int iphi = 0; iphi < num_phi; ++iphi) {
      auto c = (*(gl->cell_grid()))[ieta][iphi];
      if (c) {
        cells[ieta * num_phi + iphi] = c->calo_hash();
        ++ncells;
      } else {
        cells[ieta * num_phi + iphi] = -1;
        ++empty_cells;
      }
    }  // Loop over phi
  }    // Loop over eta
}  // copy_geo_region

// void test_device_cells(cl::sycl::queue queue) {
// // CUDA does not support experimental::printf()
// #ifndef USE_PI_CUDA
//   std::cout << "Test device cells..." << std::endl;
//   auto ev_cellinfo = queue.submit([&](cl::sycl::handler& cgh) {
//     cgh.parallel_for<class Dummy>(
//         cl::sycl::range<1>(num_cells_),
//         [=, dev_cells_local = this->device_cells_](cl::sycl::id<1> idx) {
//           unsigned int id = (int)idx[0];
//           if ((id + 1) % 10000 == 0) {
//             long long cell_id = dev_cells_local[id].identify();
//             unsigned long long hash = dev_cells_local[id].calo_hash();
//             cl::sycl::intel::experimental::printf(kCellInfo, cell_id, hash);
//           }
//         });
//   });
//   ev_cellinfo.wait_and_throw();
// #else
//   std::cout << "CUDA PI does not support experimental::printf(); "
//             << "cannot execute `test_device_cells()`." << std::endl;
//   return;
// #endif
// } // test_device_cells

// main
int main() {
  std::cout << "*** Sycl_test BEGINS ***" << std::endl;

  // Load the geometry
  CaloGeometryFromFile* calo_geo = get_calo_geo();

  // Allocate host memory for geometry samplings and regions
  unsigned int num_regions = calo_geo->get_tot_regions();
  GeoRegion* geo_regions = (GeoRegion*)malloc(num_regions * sizeof(GeoRegion));
  assert(geo_regions);
  SampleIndex* region_si =
      (SampleIndex*)malloc(CaloGeometry::MAX_SAMPLING * sizeof(SampleIndex));

  // Prepare device geometry.
  // Analogous to TFCSShapeValidation::GeoL()
  Geo* geo = new Geo();
  geo->set_num_cells(calo_geo->get_cells()->size());
  geo->set_max_sample(CaloGeometry::MAX_SAMPLING);
  geo->set_num_regions(num_regions);
  geo->set_cell_map(calo_geo->get_cells());
  geo->set_host_regions(geo_regions);
  geo->set_sample_index(region_si);

  // Copy geometry from CaloGeometryFromFile
  // Loop over samples
  unsigned int i = 0;
  for (int isamp = 0; isamp < CaloGeometry::MAX_SAMPLING; ++isamp) {
    // Get the number of regions in this sampling
    unsigned int num_regions = calo_geo->get_n_regions(isamp);

    // Assign sampling information
    region_si[isamp].index = i;
    region_si[isamp].size = num_regions;
    // Loop over regions
    for (unsigned int ireg = 0; ireg < num_regions; ++ireg) {
      copy_geo_region(calo_geo->get_region(isamp, ireg), &geo_regions[i++]);
    }  // Loop over regions
  }    // Loop over samples

  // Load geometry to device.
  bool load_success = geo->LoadDeviceGeo();
  if (!load_success) {
    std::cout << "Could not load device geometry!" << std::endl;
    return -1;
  }

  std::cout << "*** Sycl_test ENDS ***" << std::endl;
  return 0;
}  // main