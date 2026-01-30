/**
 * @file exp_base.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Base class for experiment
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "exp_config.h"
#include "exp_io.hpp"
#include "exp_utils.h"

#include <pcl/filters/voxel_grid.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <glog/logging.h>

#include <string>

namespace demo {

template <typename FeatureSignature> class ExpBase {
  public:
    using FeatureCloud = typename pcl::PointCloud<FeatureSignature>;
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    ExpBase() = default;
    virtual ~ExpBase() = default;

    virtual int evaluate(const std::string& data_path, const RealDataConfig& config) = 0;
};

template <typename FeatureSignature> class FeatWizardBase {
  public:
    using FeatureCloud = typename pcl::PointCloud<FeatureSignature>;
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    FeatWizardBase() = default;
    virtual ~FeatWizardBase() = default;

    virtual int getPointCloudandFeatures(const std::string& prefix, const RealDataConfig& config,
                                         pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, FeatureCloudPtr& features) = 0;
};

template <typename FeatureSignature> class FeatWizardHandCraft : public FeatWizardBase<FeatureSignature> {
  public:
    using FeatureCloud = typename pcl::PointCloud<FeatureSignature>;
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    FeatWizardHandCraft() = default;
    ~FeatWizardHandCraft() override = default;

    int getPointCloudandFeatures(const std::string& prefix, const RealDataConfig& config,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, FeatureCloudPtr& features) override;
};

template <typename FeatureSignature> class FeatWizardProcessed : public FeatWizardBase<FeatureSignature> {
  public:
    using FeatureCloud = typename pcl::PointCloud<FeatureSignature>;
    using FeatureCloudPtr = typename pcl::PointCloud<FeatureSignature>::Ptr;
    FeatWizardProcessed() = default;
    ~FeatWizardProcessed() override = default;

    int getPointCloudandFeatures(const std::string& prefix, const RealDataConfig& config,
                                 pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, FeatureCloudPtr& features) override;
};

/*****************FeatWizardHandCraft******************/
template <typename FeatureSignature>
int FeatWizardHandCraft<FeatureSignature>::getPointCloudandFeatures(const std::string& prefix,
                                                                    const RealDataConfig& config,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                                    FeatureCloudPtr& features) {
    if (config.feature != eFPFH) {
        LOG(ERROR) << "Other handcraft features are not implemented, please use FPFH";
        return -1;
    }
    std::string suffix = config.data_set == eKITTI ? ".bin" : ".ply";
    std::string cloud_file_name = prefix + suffix;
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_origin(new pcl::PointCloud<pcl::PointXYZ>);
    if (demo::readPointCloud<pcl::PointXYZ>(cloud_file_name, cloud_origin) < 0) {
        LOG(ERROR) << "Failed to read cloud: " << cloud_file_name;
        return -1;
    }

    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter;
    voxel_filter.setLeafSize(config.voxel_size, config.voxel_size, config.voxel_size);
    voxel_filter.setInputCloud(cloud_origin);
    voxel_filter.filter(*cloud);

    demo::computeFPFH(cloud, features, config.radius_normal, config.radius_feature);
    return 0;
}
/*****************FeatWizardHandCraft end******************/

/*****************FeatWizardProcessed******************/
template <typename FeatureSignature>
int FeatWizardProcessed<FeatureSignature>::getPointCloudandFeatures(const std::string& prefix,
                                                                    const RealDataConfig& config,
                                                                    pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                                    FeatureCloudPtr& features) {
    std::string suffix;
    std::string suffix_feature;
    switch (config.feature) {
    case eFPFH:
        suffix = "_fpfh_points.csv";
        suffix_feature = "_fpfh_features.csv";
        break;
    case eFCGF:
        suffix = "_fcgf_points.csv";
        suffix_feature = "_fcgf_features.csv";
        break;
    case ePredator:
        suffix = "_predator_points.csv";
        suffix_feature = "_predator_features.csv";
        break;
    default:
        LOG(ERROR) << "Invalid feature type";
        return -1;
    }

    std::string cloud_file_name = prefix + suffix;
    std::string feature_file_name = prefix + suffix_feature;
    if (demo::readPointCloud<pcl::PointXYZ>(cloud_file_name, cloud) < 0) {
        LOG(ERROR) << "Failed to read cloud: " << cloud_file_name;
        return -1;
    }
    if (demo::readFeatureCloud<FeatureSignature>(feature_file_name, features, false) < 0) {
        LOG(ERROR) << "Failed to read cloud: " << cloud_file_name;
        return -1;
    }
    return 0;
}
/*****************FeatWizardProcessed end******************/

} // namespace demo
