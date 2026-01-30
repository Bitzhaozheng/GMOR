#pragma once

#include "exp/exp_config.h"

namespace demo {

// Simulation
int demo_simulation(const SimDataConfig& config);
// Real data
int demo_realdata(const RealDataConfig& config, bool icp_refine = false);

// Run experiment of 3DMatch and KITTI dataset
int run_exp(const std::string& data_path, const RealDataConfig& config);

} // namespace demo
