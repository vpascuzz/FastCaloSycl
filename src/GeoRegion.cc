//
// GeoRegion.cc
//

#include "GeoRegion.h"

#include <algorithm>

#define PI 3.14159265358979323846
#define TWOPI 2 * 3.14159265358979323846

double Phi_mpi_pi(double x) {
  // TODO: Check for NaN
  while (x >= PI) {
    x -= TWOPI;
  }
  while (x < -PI) {
    x += TWOPI;
  }
  return x;
}

GeoRegion::GeoRegion()
    : cell_grid_(nullptr),
      cells_(nullptr),
      index_(0),
      cell_grid_eta_(0),
      cell_grid_phi_(0),
      xy_grid_adjust_(0.0),
      deta_(0.0),
      dphi_(0.0),
      min_eta_(0.0),
      min_phi_(0.0),
      max_eta_(0.0),
      max_phi_(0.0),
      min_eta_raw_(0.0),
      min_phi_raw_(0.0),
      max_eta_raw_(0.0),
      max_phi_raw_(0.0),
      eta_corr_(0.0),
      phi_corr_(0.0),
      min_eta_corr_(0.0),
      max_eta_corr_(0.0),
      min_phi_corr_(0.0),
      max_phi_corr_(0.0) {}

bool GeoRegion::index_range_adjust(int& ieta, int& iphi) {
  while (iphi < 0) {
    iphi += cell_grid_phi_;
  }
  while (iphi >= cell_grid_phi_) {
    iphi -= cell_grid_phi_;
  }
  if (ieta < 0) {
    ieta = 0;
    return false;
  }
  if (ieta > cell_grid_eta_) {
    ieta = cell_grid_eta_ - 1;
    return false;
  }
  return true;
}

float GeoRegion::calc_distance_eta_phi(const long long dde, float eta,
                                       float phi, float& dist_eta0,
                                       float& dist_phi0) {
  dist_eta0 = (eta - cells_[dde].eta()) / deta_double_;
  dist_phi0 = (Phi_mpi_pi(phi - cells_[dde].phi())) / dphi_double_;
  float abs_dist_eta0 = cl::sycl::abs(dist_eta0);
  float abs_dist_phi0 = cl::sycl::abs(dist_phi0);
  return cl::sycl::max(abs_dist_eta0, abs_dist_phi0) - 0.5;
}

// get_cell
long long GeoRegion::get_cell(float eta, float phi, float* distance,
                              unsigned int* steps) {
  float dist = 0.0;
  long long best_dde = -1;
  if (!distance) {
    distance = &dist;
  }
  (*distance) = +10000000;

  unsigned int intsteps = 0;
  if (!steps) {
    steps = &intsteps;
  }

  float best_eta_corr = eta_corr_;
  float best_phi_corr = phi_corr_;

  float raw_eta = eta + best_eta_corr;
  float raw_phi = eta + best_phi_corr;

  int ieta = raw_eta_pos_to_index(raw_eta);
  int iphi = raw_phi_pos_to_index(raw_phi);
  index_range_adjust(ieta, iphi);

  long long new_dde = cell_grid_[ieta * cell_grid_phi_ + iphi];
  float best_dist = +10000000;
  ++(*steps);
  unsigned int num_search = 0;
  while ((new_dde >= 0) && num_search < 3) {
    float dist_eta0 = 0.0;
    float dist_phi0 = 0.0;
    (*distance) =
        calc_distance_eta_phi(new_dde, eta, phi, dist_eta0, dist_phi0);
    best_dde = new_dde;
    best_dist = (*distance);

    if ((*distance) < 0) {
      return new_dde;
    }

    // Correct eta and phi indices by the observed difference to the cell that
    // was hit.
    ieta += cl::sycl::round(dist_eta0);
    iphi += cl::sycl::round(dist_phi0);
    index_range_adjust(ieta, iphi);
    long long old_dde = new_dde;
    new_dde = cell_grid_[ieta * cell_grid_phi_ + iphi];
    ++(*steps);
    ++num_search;
    if (old_dde == new_dde) {
      break;
    }
  }
  float min_ieta = ieta + cl::sycl::floor(min_eta_corr_ / cell_grid_eta_);
  float min_iphi = iphi + cl::sycl::floor(min_phi_corr_ / cell_grid_phi_);
  float max_ieta = ieta + cl::sycl::ceil(max_eta_corr_ / cell_grid_eta_);
  float max_iphi = iphi + cl::sycl::ceil(max_phi_corr_ / cell_grid_phi_);

  if (min_ieta < 0) {
    min_ieta = 0;
  }
  if (max_ieta > cell_grid_eta_) {
    max_ieta = cell_grid_eta_ - 1;
  }
  for (int iieta = min_ieta; iieta <= max_ieta; ++iieta) {
    for (int iiphi = min_iphi; iiphi <= max_iphi; ++iiphi) {
      ieta = iieta;
      iphi = iiphi;
      index_range_adjust(ieta, iphi);
      new_dde = cell_grid_[ieta * cell_grid_phi_ + iphi];
      ++(*steps);

      if (new_dde >= 0) {
        float dist_eta0 = 0.0;
        float dist_phi0 = 0.0;
        (*distance) =
            calc_distance_eta_phi(new_dde, eta, phi, dist_eta0, dist_phi0);

        if ((*distance) < 0) {
          return new_dde;
        } else if ((*distance) < best_dist) {
          best_dde = new_dde;
          best_dist = (*distance);
        }
      } else {
        // exception
      }
    }
  }
  (*distance) = best_dist;
  return best_dde;
}
