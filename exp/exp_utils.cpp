/**
 * @file exp_utils.cpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#include "exp_utils.h"

#include "exp_3DMatch.hpp"
#include "exp_KITTI.hpp"

#include "gmor.h"
#include "trde.h"

#include <pcl/features/fpfh_omp.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/registration/ia_ransac.h>

#include <glog/logging.h>

#include <chrono>

namespace demo {

typedef pcl::PointCloud<gmor::FeatureDescriptor<33>> FPFHCloud;
typedef pcl::PointCloud<gmor::FeatureDescriptor<33>>::Ptr FPFHCloudPtr;

void computeFPFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs,
                 double radius_normal, double radius_feature, bool use_sensor_origin,
                 const Eigen::Vector3f& viewpoint) {
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    computeFPFH(cloud, normals, FPFHs, radius_normal, radius_feature, use_sensor_origin, viewpoint);
}

void computeFPFH(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                 pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_normal, double radius_feature,
                 bool use_sensor_origin, const Eigen::Vector3f& viewpoint) {
    pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> normal_estimation;
    normal_estimation.setRadiusSearch(radius_normal);
    if (use_sensor_origin)
        normal_estimation.useSensorOriginAsViewPoint();
    else
        normal_estimation.setViewPoint(viewpoint[0], viewpoint[1], viewpoint[2]);
    normal_estimation.setInputCloud(cloud);
    normal_estimation.compute(*normals);
    computeFPFHwithNormals(cloud, normals, FPFHs, radius_feature);
}

void computeFPFHwithNormals(const pcl::PointCloud<pcl::PointXYZINormal>::Ptr& cloud_normals,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_feature) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
    for (auto& point_normal : *cloud_normals) {
        cloud->emplace_back(point_normal.x, point_normal.y, point_normal.z);
        normals->emplace_back(point_normal.normal_x, point_normal.normal_y, point_normal.normal_z);
    }
    computeFPFHwithNormals(cloud, normals, FPFHs, radius_feature);
}

void computeFPFHwithNormals(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                            const pcl::PointCloud<pcl::Normal>::Ptr& normals,
                            pcl::PointCloud<pcl::FPFHSignature33>::Ptr& FPFHs, double radius_feature) {
    pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh_estimation;
    fpfh_estimation.setRadiusSearch(radius_feature);
    fpfh_estimation.setInputCloud(cloud);
    fpfh_estimation.setInputNormals(normals);
    fpfh_estimation.compute(*FPFHs);
}

void computeFPFHCorrs(const pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
                      const pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud, pcl::Correspondences& correspondences,
                      const RealDataConfig& config, bool use_sensor_origin_target, const Eigen::Vector3f& viewpoint) {
    // Estimate FPFH features
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr source_features(new pcl::PointCloud<pcl::FPFHSignature33>);
    pcl::PointCloud<pcl::FPFHSignature33>::Ptr target_features(new pcl::PointCloud<pcl::FPFHSignature33>);

    computeFPFH(source_cloud, source_features, config.radius_normal, config.radius_feature);
    computeFPFH(target_cloud, target_features, config.radius_normal, config.radius_feature, use_sensor_origin_target,
                viewpoint);

    matchFPFH(source_features, target_features, correspondences, config, false);
}

void matchFPFH(const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_features,
               const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_features, pcl::Correspondences& corrs,
               const RealDataConfig& config, bool use_selfmade_descriptors) {
    if (use_selfmade_descriptors) {
        FPFHCloudPtr src_descriptors(new FPFHCloud);
        FPFHCloudPtr tgt_descriptors(new FPFHCloud);
        for (auto& feature : *source_features) {
            gmor::FeatureDescriptor<33> descriptor;
            std::copy(feature.histogram, feature.histogram + 33, descriptor.histogram);
            src_descriptors->push_back(descriptor);
        }
        for (auto& feature : *target_features) {
            gmor::FeatureDescriptor<33> descriptor;
            std::copy(feature.histogram, feature.histogram + 33, descriptor.histogram);
            tgt_descriptors->push_back(descriptor);
        }
        // Here are two ways to initialize the matcher: Constructor, or setNeighbors() and setdf()
        gmor::FeatureMatcher<gmor::FeatureDescriptor<33>, gmor::KdTreeFLANN<gmor::FeatureDescriptor<33>>> matcher(
            config.knn, config.df);
        matcher.match(src_descriptors, tgt_descriptors, corrs, gmor::eCross | gmor::eSoftmax);
    } else {
        // pcl::KdTreeFLANN thread unsafe?
        gmor::FeatureMatcher<pcl::FPFHSignature33, pcl::KdTreeFLANN<pcl::FPFHSignature33>> matcher;
        matcher.setNeighbors(config.knn);
        matcher.setdf(config.df);
        matcher.match(source_features, target_features, corrs, gmor::eCross | gmor::eSoftmax);
    }
}

double globalRegistrationRANSAC_PCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                                    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& source_features,
                                    const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt,
                                    const pcl::PointCloud<pcl::FPFHSignature33>::Ptr& target_features,
                                    const RealDataConfig& config, Eigen::Matrix4f& output) {
    // RANSAC, Default error function is TruncatedError (error = min(dist / corr_dist_threshold_, 1.0))
    pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
    sac_ia.setInputSource(src);
    sac_ia.setSourceFeatures(source_features);
    sac_ia.setInputTarget(tgt);
    sac_ia.setTargetFeatures(target_features);
    // Threshold of error function TruncatedError, default is 100.0 for numerical stability
    sac_ia.setMaxCorrespondenceDistance(config.noise_bound);
    // Minimum distance between two samples, default is 0
    sac_ia.setMinSampleDistance(config.noise_bound);
    // Samples in each iteration, default is 3
    sac_ia.setNumberOfSamples(3);
    // Maximum number of RANSAC iterations, default is 10
    sac_ia.setMaximumIterations(config.top_k); // Reuse top_k parameter
    // Calculate k nearest neighbors in the feature space (default is 10), and sample a random point as correspondence
    sac_ia.setCorrespondenceRandomness(config.knn);
    pcl::PointCloud<pcl::PointXYZ>::Ptr src_trans(new pcl::PointCloud<pcl::PointXYZ>);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    sac_ia.align(*src_trans);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    sac_ia.getFitnessScore();
    output = sac_ia.getFinalTransformation();

    return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count() * 1e-3;
}

double globalRegistrationPCL(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                             const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt, const pcl::Correspondences& corrs,
                             const RegConfig& config, Eigen::Matrix4f& output) {
    // Extract correspondenses and transform to Eigen::Matrix3Xf
    std::vector<std::tuple<int, int, float>> corrs_input;
    corrs_input.reserve(corrs.size());
    Eigen::Matrix3Xf src_points = src->getMatrixXfMap().topRows<3>();
    Eigen::Matrix3Xf tgt_points = tgt->getMatrixXfMap().topRows<3>();

    for (const auto& corr : corrs) {
        corrs_input.emplace_back(corr.index_query, corr.index_match, corr.weight);
    }

    return globalRegistration(src_points, tgt_points, corrs_input, config, output);
}

double globalRegistrationPCL_Axis(const pcl::PointCloud<pcl::PointXYZ>::Ptr& src,
                                  const pcl::PointCloud<pcl::PointXYZ>::Ptr& tgt, const pcl::Correspondences& corrs,
                                  const RegConfig& config, Eigen::Matrix4f& output, const Eigen::Vector3f& axis) {
    // Extract correspondenses and transform to Eigen::Matrix3Xf
    std::vector<std::tuple<int, int, float>> corrs_input;
    corrs_input.reserve(corrs.size());
    Eigen::Matrix3Xf src_points = src->getMatrixXfMap().topRows<3>();
    Eigen::Matrix3Xf tgt_points = tgt->getMatrixXfMap().topRows<3>();

    for (const auto& corr : corrs) {
        corrs_input.emplace_back(corr.index_query, corr.index_match, corr.weight);
    }

    return globalRegistration_Axis(src_points, tgt_points, corrs_input, config, output, axis);
}

double globalRegistration(const Eigen::Matrix3Xf& src_points, const Eigen::Matrix3Xf& tgt_points,
                          const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                          Eigen::Matrix4f& output) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    gmor::RegistrationBnBBase<float>::Ptr_u reg;
    if (config.method == eTRDE) {
        trde::TRDESolver::Ptr_u trde_reg(new trde::TRDESolver);
        reg = std::move(trde_reg);
        // Normalization is required due to translation search
        reg->setNormalize(true);
    } else if (config.method == eGMOR) {
        gmor::GMOSolver::Ptr_u gmor_reg(new gmor::GMOSolver);
        gmor_reg->setTopk(config.top_k);
        gmor_reg->setRho(config.rho);
        gmor_reg->setRotNearZ(config.rot_near_z); // For KITTI dataset
        reg = std::move(gmor_reg);
        reg->setNormalize(false);
    } else {
        LOG(ERROR) << "Invalid registration method: " << config.method;
        return -1;
    }
    reg->setNoiseBound(config.noise_bound);
    reg->setNumThreads(config.num_threads);
    reg->setBranchThreshold(config.branch_eps);
    reg->setBoundThreshold(config.bound_eps);
    output = reg->solve(src_points, tgt_points, corrs);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count() * 1e-3;
}

double globalRegistration_Axis(const Eigen::Matrix3Xf& src_points, const Eigen::Matrix3Xf& tgt_points,
                               const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                               Eigen::Matrix4f& output, const Eigen::Vector3f& axis) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    gmor::GMOSolver::Ptr_u reg(new gmor::GMOSolver);
    reg->setTopk(config.top_k);
    reg->setRho(config.rho);
    reg->setRotNearZ(config.rot_near_z); // For KITTI dataset
    reg->setNormalize(false);

    reg->setNoiseBound(config.noise_bound);
    reg->setNumThreads(config.num_threads);
    reg->setBranchThreshold(config.branch_eps);
    reg->setBoundThreshold(config.bound_eps);
    output = reg->solvewithAxis(src_points, tgt_points, corrs, axis);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count() * 1e-3;
}

double globalRegistration2D(const Eigen::Matrix2Xf& src_points, const Eigen::Matrix2Xf& tgt_points,
                            const std::vector<std::tuple<int, int, float>>& corrs, const RegConfig& config,
                            Eigen::Matrix3f& output) {
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();

    gmor::GMOSolver::Ptr_u reg(new gmor::GMOSolver);
    reg->setTopk(config.top_k);
    reg->setRho(config.rho);
    reg->setRotNearZ(config.rot_near_z); // For KITTI dataset
    reg->setNormalize(false);

    reg->setNoiseBound(config.noise_bound);
    reg->setNumThreads(config.num_threads);
    reg->setBranchThreshold(config.branch_eps);
    reg->setBoundThreshold(config.bound_eps);
    output = reg->solve2D(src_points, tgt_points, corrs);

    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::duration<int, std::micro>>(end - start).count() * 1e-3;
}

int exp_3DMatch(const std::string& data_path, const RealDataConfig& config) {
    switch (config.feature) {
    case FeatureType::eFPFH:
        return exp_3DMatch_FPFH(data_path, config);
    case FeatureType::eFCGF:
    case FeatureType::ePredator:
        return exp_3DMatch_FCGF(data_path, config);
    default:
        LOG(ERROR) << "Invalid feature type: " << config.feature;
        return -1;
    }
}

int exp_KITTI(const std::string& data_path, const RealDataConfig& config) {
    switch (config.feature) {
    case FeatureType::eFPFH:
        return exp_KITTI_FPFH(data_path, config);
    case FeatureType::eFCGF:
        return exp_KITTI_FCGF(data_path, config);
    default:
        LOG(ERROR) << "Invalid feature type: " << config.feature;
        return -1;
    }
}

int exp_3DMatch_FPFH(const std::string& data_path, const RealDataConfig& config) {
    Exp3DMatchHandCraft<pcl::FPFHSignature33> experiment;
    experiment.parseDataSets(config);
    return experiment.evaluate(data_path, config);
}

int exp_3DMatch_FCGF(const std::string& data_path, const RealDataConfig& config) {
    Exp3DMatchProcessed<gmor::FeatureDescriptor<32>> experiment;
    experiment.parseDataSets(config);
    return experiment.evaluate(data_path, config);
}

int exp_KITTI_FPFH(const std::string& data_path, const RealDataConfig& config) {
    ExpKITTIHandCraft<pcl::FPFHSignature33> experiment;
    return experiment.evaluate(data_path, config);
}

int exp_KITTI_FCGF(const std::string& data_path, const RealDataConfig& config) {
    ExpKITTIProcessed<gmor::FeatureDescriptor<32>> experiment;
    return experiment.evaluate(data_path, config);
}

float normalizePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud) {
    // Find centers and bounding boxes
    pcl::PointXYZ source_center, source_min_pt, source_max_pt, target_min_pt, target_max_pt;
    pcl::getMinMax3D(*source_cloud, source_min_pt, source_max_pt);
    source_center.x = 0.5f * (source_min_pt.x + source_max_pt.x);
    source_center.y = 0.5f * (source_min_pt.y + source_max_pt.y);
    source_center.z = 0.5f * (source_min_pt.z + source_max_pt.z);

    // Max length of bounding boxes
    float max_length = 0.0;
    for (int i = 0; i < 3; ++i) {
        float length =
            std::max(source_max_pt.data[i] - source_min_pt.data[i], target_max_pt.data[i] - target_min_pt.data[i]);
        if (length > max_length) {
            max_length = length;
        }
    }

    // Scaling source and target clouds to [-1, 1] cube
    float scale = 2.0f / max_length;
    for (auto& p : source_cloud->points) {
        p.x = (p.x - source_center.x) * scale;
        p.y = (p.y - source_center.y) * scale;
        p.z = (p.z - source_center.z) * scale;
    }
    return scale;
}

void addNoiseandOutliers(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, double noise_bound, double outlier_ratio,
                         double outlier_bound_min, double outlier_bound_max) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::normal_distribution<> noise_dist(0, noise_bound);
    std::uniform_real_distribution<> outlier_p(0, 1);
    std::uniform_real_distribution<> outlier_dist(2 * outlier_bound_min - outlier_bound_max, outlier_bound_max);
    for (auto& p : cloud->points) {
        if (outlier_p(rng) < outlier_ratio) {
            Eigen::Vector3f outlier_vec;
            genRandTranslation(outlier_vec, eSPHERE, outlier_bound_min, outlier_bound_max);
            p.x = outlier_vec.x();
            p.y = outlier_vec.y();
            p.z = outlier_vec.z();
        }
        p.x += noise_dist(rng);
        p.y += noise_dist(rng);
        p.z += noise_dist(rng);
    }
}

void genRandRotMat(Eigen::Matrix3f& R) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_real_distribution<float> dist(-1.0, 1.0);

    // Random unit vector
    Eigen::Vector3f rot_axis;
    rot_axis.z() = dist(rng);
    float r = std::sqrt(1 - rot_axis.z() * rot_axis.z());
    float theta = M_PI * dist(rng);
    rot_axis.x() = r * std::cos(theta);
    rot_axis.y() = r * std::sin(theta);

    // Random rotation angle
    float rot_angle = dist(rng) * M_PI; // [-pi, pi]
    Eigen::AngleAxisf R_aa(rot_angle, rot_axis);
    R = R_aa.toRotationMatrix();
}

void genRandTranslation(Eigen::Vector3f& t, TranslationSampleType type, float min_t, float max_t) {
    if (min_t < 0.0f) {
        min_t = 0.0f;
        LOG(WARNING) << "Min translation is negative, setting to " << min_t;
    }
    if (max_t < std::numeric_limits<float>::epsilon()) {
        max_t = std::numeric_limits<float>::epsilon();
        LOG(WARNING) << "Max translation is too small, setting to " << max_t;
    }
    if (min_t > max_t) {
        std::swap(min_t, max_t);
        LOG(WARNING) << "Min translation is greater than max translation, swapping them";
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    if (type == eCUBE) {
        std::uniform_real_distribution<float> dist(-max_t, max_t);
        // Rejection sampling to ensure that the translation is within the cube
        do {
            t.x() = dist(rng);
            t.y() = dist(rng);
            t.z() = dist(rng);
        } while (t.x() > -min_t && t.x() < min_t && t.y() > -min_t && t.y() < min_t && t.z() > -min_t && t.z() < min_t);

    } else if (type == eSPHERE) {
        float start_samp = pow(min_t / max_t, 3);
        std::uniform_real_distribution<float> dist(start_samp, 1.0);
        float r3 = dist(rng);
        float r = std::cbrt(r3) * max_t;
        t.z() = (2 * dist(rng) - start_samp - 1.0) / (1.0 - start_samp) * r;
        float r_2 = std::sqrt(r * r - t.z() * t.z());
        float theta = 2 * M_PI * (dist(rng) - start_samp) / (1.0 - start_samp);
        t.x() = r_2 * std::cos(theta);
        t.y() = r_2 * std::sin(theta);
    } else {
        LOG(ERROR) << "Invalid translation sample type: " << type;
    }
}

std::vector<int> sampleN(std::vector<int>& v, int n, int seed_or_rd) {
    return gmor::sampleN<int>(v, n, seed_or_rd, false);
}

float getRotDist(const Eigen::Matrix3f& R1, const Eigen::Matrix3f& R2, bool isDegree) {
    return gmor::getRotDist<float>(R1, R2, isDegree);
}

} // namespace demo
