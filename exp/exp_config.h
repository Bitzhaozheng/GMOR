/**
 * @file exp_config.h
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Configuration for experiments.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <string>

namespace demo {

enum DataSet { e3DMatch = 0, e3DLoMatch = 1, eKITTI = 2 };

enum FeatureType { eFPFH = 0, eFCGF = 1, ePredator = 2 };

enum RegistrationMethod { eTRDE = 0, eGMOR = 1 };

struct RegConfig {
    std::string src_file_name;
    RegistrationMethod method = eGMOR;
    float noise_bound = 0.10;
    int num_threads = 12;
    int num_samples = 8000;
    float branch_eps = 5e-2;
    float bound_eps = 1e-3;
    int top_k = 12;
    float rho = 0.25;
    bool rot_near_z = false; // For KITTI dataset
};

struct SimDataConfig : RegConfig {
    explicit SimDataConfig(const RegConfig& config) : RegConfig(config) {};
    double outlier_ratio = 0.95;
};

// Default for 3DMatch/3DLoMatch dataset with FPFH feature
struct RealDataConfig : RegConfig {
    explicit RealDataConfig(const RegConfig& config) : RegConfig(config) {};
    std::string tgt_file_name;
    double voxel_size = 0.05;
    double radius_normal = 0.10;
    double radius_feature = 0.30;
    DataSet data_set = e3DMatch;
    FeatureType feature = eFPFH;
    int knn = 40;
    float df = 0.01;
    bool random_sample = false;
};

} // namespace demo
