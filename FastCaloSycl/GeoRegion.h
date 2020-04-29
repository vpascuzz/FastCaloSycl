//
// GeoRegion.h
//

#ifndef FASTCALOSYCL_GEOREGION_H_
#define FASTCALOSYCL_GEOREGION_H_

#include <CL/sycl.hpp>
#include <memory>

#include "CaloDetDescrElement.h"

class GeoRegion {
 public:
  // Constructors
  GeoRegion();
  // Destructor
  ~GeoRegion() { free(cell_grid_); }

  inline void set_cells(CaloDetDescrElement* dde) { cells_ = dde; }
  inline void set_cells_device(CaloDetDescrElement* dde) {
    cells_device_ = dde;
  }
  inline void set_cell_grid(long long* cells) { cell_grid_ = cells; }
  inline void set_cell_grid_device(long long* cells) {
    cell_grid_device_ = cells;
  }
  inline void set_index(int i) { index_ = i; }
  inline void set_cell_grid_eta(int eta) { cell_grid_eta_ = eta; }
  inline void set_cell_grid_phi(int phi) { cell_grid_phi_ = phi; }
  inline void set_xy_grid_adjust(float adjust) { xy_grid_adjust_ = adjust; }
  inline void set_deta(float deta) { deta_ = deta; }
  inline void set_dphi(float dphi) { dphi_ = dphi; }
  inline void set_min_eta(float min_eta) { min_eta_ = min_eta; }
  inline void set_min_phi(float min_phi) { min_phi_ = min_phi; }
  inline void set_max_eta(float max_eta) { max_eta_ = max_eta; }
  inline void set_max_phi(float max_phi) { max_phi_ = max_phi; }
  inline void set_min_eta_raw(float min_eta) { min_eta_raw_ = min_eta; }
  inline void set_min_phi_raw(float min_phi) { min_phi_raw_ = min_phi; }
  inline void set_max_eta_raw(float max_eta) { max_eta_raw_ = max_eta; }
  inline void set_max_phi_raw(float max_phi) { max_phi_raw_ = max_phi; }
  inline void set_eta_corr(float eta_corr) { eta_corr_ = eta_corr; }
  inline void set_phi_corr(float phi_corr) { phi_corr_ = phi_corr; }
  inline void set_min_eta_corr(float eta_corr) { min_eta_corr_ = eta_corr; }
  inline void set_min_phi_corr(float phi_corr) { min_phi_corr_ = phi_corr; }
  inline void set_max_eta_corr(float eta_corr) { max_eta_corr_ = eta_corr; }
  inline void set_max_phi_corr(float phi_corr) { max_phi_corr_ = phi_corr; }
  // Not sure what "double" means.
  inline void set_deta_double(float deta) { deta_double_ = deta; }
  inline void set_dphi_double(float dphi) { dphi_double_ = dphi; }

  inline long long* cell_grid() { return cell_grid_; }
  inline CaloDetDescrElement* cells() { return cells_; }
  inline int index() { return index_; }
  inline int cell_grid_eta() { return cell_grid_eta_; }
  inline int cell_grid_phi() { return cell_grid_phi_; }
  inline float min_eta() { return min_eta_; };
  inline float min_phi() { return min_phi_; };
  inline float max_eta() { return max_eta_; };
  inline float max_phi() { return max_phi_; };
  inline float deta() { return deta_; };
  inline float dphi() { return dphi_; };
  inline int raw_eta_pos_to_index(float eta_raw) const {
    return cl::sycl::floor((eta_raw - min_eta_raw_) / deta_double_);
  }
  inline int raw_phi_pos_to_index(float phi_raw) const {
    return cl::sycl::floor((phi_raw - min_phi_raw_) / dphi_double_);
  }

  bool index_range_adjust(int& ieta, int& iphi);
  float calc_distance_eta_phi(const long long dde, float eta, float phi,
                              float& dist_eta0, float& dist_phi0);
  long long get_cell(float eta, float phi, float* distance = nullptr,
                     unsigned int* steps = nullptr);

 private:
  long long* cell_grid_;               // Array for calorimeter cells
  long long* cell_grid_device_;        // Array for calorimeter cells on device
  CaloDetDescrElement* cells_;         // Array for detector elements
  CaloDetDescrElement* cells_device_;  // Array for detector elements on device

  int index_;
  int cell_grid_eta_;
  int cell_grid_phi_;
  float xy_grid_adjust_;
  float deta_;
  float dphi_;
  float min_eta_;
  float min_phi_;
  float max_eta_;
  float max_phi_;
  float min_eta_raw_;
  float min_phi_raw_;
  float max_eta_raw_;
  float max_phi_raw_;
  float eta_corr_;
  float phi_corr_;
  float min_eta_corr_;
  float max_eta_corr_;
  float min_phi_corr_;
  float max_phi_corr_;
  float deta_double_;  // What is "double"?
  float dphi_double_;  // ...
};

#endif  // FASTCALOSYCL_GEOREGION_H_
