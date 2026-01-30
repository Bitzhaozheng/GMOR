/**
 * @file exp_utils.h
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Utility functions for experiments and pcl interface of registration.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "exp_config.h"

#include <pcl/correspondence.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

#include <tuple>
#include <vector>

namespace demo {

/**
 * @brief Compute FPFH features between two point clouds
 *
 * @param cloud Point cloud from scanning device or others
 * @param FPFHs Output FPFH features
 * @param radius_normal Radius for normal estimation
 * @param radius_feature Radius for FPFH estimation
 * @param use_sensor_origin Whether to use pcl::PointCloud<>::sensor_origin_ as target viewpoint.
 * @param viewpoint Viewpoint of sensor in normal estimation.
 *
 * Specifically, if point cloud is not from scanning device or the normals are not estimated, the points should be
 * away from the viewpoint and tiled on the perspective plane. Otherwise, the normals may be estimated incorrectly,
 * resulting in failed FPFH matching.
 *
 */
void computeFPFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs,
                 double radius_normal = 0.05, double radius_feature = 0.1, bool use_sensor_origin = true,
                 const Eigen::Vector3f& viewpoint = Eigen::Vector3f::Zero());
void computeFPFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_normal = 0.05,
                 double radius_feature = 0.1, bool use_sensor_origin = true,
                 const Eigen::Vector3f& viewpoint = Eigen::Vector3f::Zero());

// Compute FPFH features with existing normals
void computeFPFHwithNormals(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_normals,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_feature = 0.1);
void computeFPFHwithNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                            const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_feature = 0.1);

// Compute FPFH and match
void computeFPFHCorrs(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::Correspondences& correspondences,
                      const RealDataConfig& config, bool use_sensor_origin_target = true,
                      const Eigen::Vector3f& viewpoint = Eigen::Vector3f::Zero());

// Match FPFH correspondences using nearest neighbor search
void matchFPFH(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_features,
               const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_features, pcl::Correspondences& corrs,
               const RealDataConfig& config, bool use_selfmade_descriptors = true);

// SAC_IA method in FPFH paper but too slow, recommend to use Open3D python implementation instead in experiments.
double globalRegistrationRANSAC_PCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                                    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_features,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt,
                                    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_features,
                                    const RealDataConfig& config, Eigen::Matrix4f& output);

/**
 * @brief pcl interface for global registration with estimated correspondences.
 *
 * @param src Source point cloud
 * @param tgt Target point cloud
 * @param corrs Weighted correspondences between source and target point clouds
 * @param config
 * @param output Homogeneous rigid transformation matrix
 * @return double Elapsed time in milliseconds
 */
double globalRegistrationPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt, const pcl::Correspondences& corrs,
                             const RegConfig& config, Eigen::Matrix4f& output);
double globalRegistrationPCL_Axis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt, const pcl::Correspondences& corrs,
                                  const RegConfig& config, Eigen::Matrix4f& output, const Eigen::Vector3f& axis);

/**
 * @brief Weighted correspondence-based global registration
 *
 * @param src_points Source point cloud
 * @param tgt_points Target point cloud
 * @param corrs Weighted correspondences between source and target point clouds
 * @param config
 * @param output Homogeneous rigid transformation matrix
 * @return double Elapsed time in milliseconds
 */
double globalRegistration(const Eigen::Matrix3Xf& src_points, const Eigen::Matrix3Xf& tgt_points,
                          const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                          Eigen::Matrix4f& output);
double globalRegistration_Axis(const Eigen::Matrix3Xf& src_points, const Eigen::Matrix3Xf& tgt_points,
                               const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                               Eigen::Matrix4f& output, const Eigen::Vector3f& axis);
double globalRegistration2D(const Eigen::Matrix2Xf& src_points, const Eigen::Matrix2Xf& tgt_points,
                            const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                            Eigen::Matrix3f& output);

int exp_3DMatch(const std::string& data_path, const RealDataConfig& config);
int exp_3DMatch_FPFH(const std::string& data_path, const RealDataConfig& config);
int exp_3DMatch_FCGF(const std::string& data_path, const RealDataConfig& config); // Also for Predator (32 dimensions)

int exp_KITTI(const std::string& data_path, const RealDataConfig& config);
int exp_KITTI_FPFH(const std::string& data_path, const RealDataConfig& config);
int exp_KITTI_FCGF(const std::string& data_path, const RealDataConfig& config);

// Normalize to [-1, 1]
float normalizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud);

void addNoiseandOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double noise_bound = 0.01,
                         double outlier_ratio = 0.5, double outlier_bound_min = 0.5, double outlier_bound_max = 1.0);

void genRandRotMat(Eigen::Matrix3f& R);

enum TranslationSampleType {
    eCUBE = 0,  // Generated in a cube with range [-max_t, max_t]
    eSPHERE = 1 // Generated in a sphere with radius max_t
};
void genRandTranslation(Eigen::Vector3f& t, TranslationSampleType type = eSPHERE, float min_t = 0.0, float max_t = 1.0);

std::vector<int> sampleN(std::vector<int>& v, int n, int seed_or_rd = -1);
float getRotDist(const Eigen::Matrix3f& R1, const Eigen::Matrix3f& R2, bool isDegree);

} // namespace demo
