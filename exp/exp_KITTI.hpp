/**
 * @file exp_KITTI.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Entire pipeline for KITTI dataset evaluation
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include "exp_base.hpp"
#include "exp_utils.h"
#include "utils_KITTI.h"

#include "matcher.hpp"
#include "registration_utils.hpp"

#include <pcl/common/common.h>
#include <pcl/registration/icp.h>

#include <glog/logging.h>

// System
#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#endif

#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

namespace demo {

template <typename FeatureSignature, class FeatureWizard> class ExpKITTI : public ExpBase<FeatureSignature> {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    using FeatureCloud = typename ExpBase<FeatureSignature>::FeatureCloud;
    using FeatureCloudPtr = typename ExpBase<FeatureSignature>::FeatureCloudPtr;

    ExpKITTI() = default;
    ~ExpKITTI() = default;

    void parseDataSets(const std::string& data_path);

    int evaluate(const std::string& data_path, const RealDataConfig& config) override;

  protected:
    // 555 pairs
    const std::array<std::string, 3> m_folder_list = {"08", "09", "10"};
    std::vector<TrajectoryKITTI> m_trajectory;
    FeatureWizard m_feat_reader;
};

template <typename FeatureSignature>
using ExpKITTIHandCraft = ExpKITTI<FeatureSignature, FeatWizardHandCraft<FeatureSignature>>;

template <typename FeatureSignature>
using ExpKITTIProcessed = ExpKITTI<FeatureSignature, FeatWizardProcessed<FeatureSignature>>;

/*****************Implementation******************/

/*****************ExpKITTI******************/

template <typename FeatureSignature, class FeatureWizard>
void ExpKITTI<FeatureSignature, FeatureWizard>::parseDataSets(const std::string& data_path) {
    m_trajectory.clear();
    m_trajectory.reserve(m_folder_list.size());
    for (auto& folder : m_folder_list) {
        TrajectoryKITTI traj;
        std::string pair_name, poses_name, calib_name;
        pair_name.append("../benchmarks/KITTI/pairs").append(folder).append(".csv");
        poses_name.append(data_path).append("/poses/").append(folder).append(".txt");
        calib_name.append(data_path).append("/sequences/").append(folder).append("/calib.txt");
        traj.LoadFromFile(pair_name, poses_name, calib_name);
        m_trajectory.push_back(traj);
    }
}

template <typename FeatureSignature, class FeatureWizard>
int ExpKITTI<FeatureSignature, FeatureWizard>::evaluate(const std::string& data_path, const RealDataConfig& config) {
    int recall_sum = 0;
    int num_frames = 0;
    // Change it for long-term consecutive evaluation
    constexpr int start_frame = 0;
    std::ofstream result_csv("./result_KITTI.csv", std::ios::out);
    result_csv << "Dataset, RE, TE, RegTime, TotalTime" << std::endl;
    this->parseDataSets(data_path);
    gmor::FeatureMatcher<FeatureSignature, gmor::KdTreeFLANN<FeatureSignature>> matcher(config.knn, config.df);
    for (size_t i = 0; i < this->m_trajectory.size(); i++) {
        TrajectoryKITTI traj = this->m_trajectory[i];
        for (const auto& frame : traj.data_) {
            if (num_frames < start_frame) {
                num_frames++;
                continue;
            }
            pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
            FeatureCloudPtr src_features(new FeatureCloud);
            FeatureCloudPtr tgt_features(new FeatureCloud);
            pcl::PointCloud<pcl::PointXYZ>::Ptr src_o(new pcl::PointCloud<pcl::PointXYZ>);
            pcl::PointCloud<pcl::PointXYZ>::Ptr tgt_o(new pcl::PointCloud<pcl::PointXYZ>);
            FeatureCloudPtr src_features_o(new FeatureCloud);
            FeatureCloudPtr tgt_features_o(new FeatureCloud);

            std::string prefix, src_prefix, tgt_prefix;
            if (config.feature == eFPFH) {
                prefix.append(data_path).append("/sequences/").append(m_folder_list[i]).append("/velodyne/");
                // 6 digits (e.g. 000001.bin)
                std::stringstream ss;
                ss << std::setfill('0') << std::setw(6) << frame.id1_;
                src_prefix.append(prefix).append(ss.str());
                ss.str("");
                ss << std::setfill('0') << std::setw(6) << frame.id2_;
                tgt_prefix.append(prefix).append(ss.str());
            } else {
                // Preprocessed FCGF
                prefix.append(data_path).append("/preprocessed/");
                src_prefix.append(prefix).append("src_").append(std::to_string(num_frames));
                tgt_prefix.append(prefix).append("tgt_").append(std::to_string(num_frames));
            }

            this->m_feat_reader.getPointCloudandFeatures(src_prefix, config, src_o, src_features_o);
            this->m_feat_reader.getPointCloudandFeatures(tgt_prefix, config, tgt_o, tgt_features_o);
            std::vector<int> src_indices(src_features_o->size());
            std::iota(src_indices.begin(), src_indices.end(), 0);
            std::vector<int> tgt_indices(tgt_features_o->size());
            std::iota(tgt_indices.begin(), tgt_indices.end(), 0);
            // Set random seed to 0 for reproductive results
            std::vector<int> src_sample_indices = gmor::sampleN(src_indices, config.num_samples, 0);
            std::vector<int> tgt_sample_indices = gmor::sampleN(tgt_indices, config.num_samples, 0);
            pcl::copyPointCloud(*src_o, src_sample_indices, *src);
            pcl::copyPointCloud(*src_features_o, src_sample_indices, *src_features);
            pcl::copyPointCloud(*tgt_o, tgt_sample_indices, *tgt);
            pcl::copyPointCloud(*tgt_features_o, tgt_sample_indices, *tgt_features);

            std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
            // Generate correspondences
            pcl::Correspondences corrs;
            matcher.match(src_features, tgt_features, corrs, gmor::eCross | gmor::eSoftmax);
            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

            LOG(INFO) << "Pair: " << frame.id1_ << " to " << frame.id2_ << ", src: " << src_features->size()
                      << ", tgt: " << tgt_features->size() << ", corrs: " << corrs.size();

            // Compare with ground truth
            Eigen::Matrix4d transform_gt = frame.transformation_;
            Eigen::Matrix4f transform_gt_float = transform_gt.cast<float>();

            std::string gt_dir = data_path + "/icp_gt";
            std::string gt_filename = gt_dir + "/" + m_folder_list[i] + "_" + std::to_string(frame.id1_) + "_" +
                                      std::to_string(frame.id2_) + "_gt.csv";
            std::ifstream gt_file(gt_filename);
            // Use cached ground truth
            if (gt_file.is_open()) {
                std::string line;
                int row = 0;
                while (std::getline(gt_file, line)) {
                    std::stringstream ss(line);
                    std::string val;
                    for (int col = 0; col < 4; col++) {
                        std::getline(ss, val, ',');
                        transform_gt_float(row, col) = std::stof(val);
                    }
                    row++;
                }
                gt_file.close();
            } else {
                // Refine the ground truth using ICP following FCGF project
                // The initial guess of ground truth may be inaccurate due to the calibration and camera pose errors
                pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
                icp.setInputSource(src_o);
                icp.setInputTarget(tgt_o);
                icp.setTransformationEpsilon(1e-6);
                icp.setMaxCorrespondenceDistance(0.6);
                icp.setMaximumIterations(200);
                icp.align(*src, transform_gt_float); // src is never used
                transform_gt_float = icp.getFinalTransformation();

#ifdef _WIN32
                _mkdir(gt_dir.c_str());
#else
                mkdir(gt_dir.c_str(), 0644);
#endif
                // Write to *.csv file
                std::ofstream gt_file_out(gt_filename);
                for (int row = 0; row < 4; row++) {
                    for (int col = 0; col < 3; col++) {
                        gt_file_out << transform_gt_float(row, col) << ",";
                    }
                    gt_file_out << transform_gt_float(row, 3) << std::endl;
                }
                gt_file_out.close();
            }

            Eigen::AngleAxisf axis_gt(transform_gt_float.topLeftCorner<3, 3>());
            Eigen::Matrix4f transform_result;
            // double reg_time_ms = demo::globalRegistrationPCL_Axis(src, tgt, corrs, config, transform_result, axis_gt.axis());
            double reg_time_ms = demo::globalRegistrationPCL(src, tgt, corrs, config, transform_result);
            double total_time_ms =
                reg_time_ms +
                std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count() * 1e-3;
            LOG(INFO) << "Registration time: " << reg_time_ms << " ms, total time: " << total_time_ms << " ms";

            auto rotation_error = gmor::getRotDist<float>(transform_result.topLeftCorner<3, 3>(),
                                                          transform_gt_float.topLeftCorner<3, 3>(), true);
            auto translation_error =
                (transform_result.topRightCorner<3, 1>() - transform_gt_float.topRightCorner<3, 1>()).norm();

            // p.s. TE of 3d_bbs (ICRA 2024) is around 1.0, so we change to 2.0 as the threshold for fair comparison
            if (rotation_error < 5 && translation_error < 2.0) {
                recall_sum++;
            } else {
                Eigen::AngleAxisf axis(transform_result.topLeftCorner<3, 3>());
                LOG(WARNING) << "Rotation axis offset: " << (axis.axis() - axis_gt.axis()).transpose();
                LOG(WARNING) << "Rotation axis error: " << (axis.axis() - axis_gt.axis()).norm();
                LOG(WARNING) << "Translation offset: "
                             << (transform_result.topRightCorner<3, 1>() - transform_gt_float.topRightCorner<3, 1>())
                                    .transpose();
            }
            LOG(INFO) << "RE: " << rotation_error << " (deg), " << "TE: " << translation_error
                      << ", RR: " << (recall_sum * 100.0) / (num_frames + 1 - start_frame) << "%";

            // Write to csv
            result_csv << num_frames << ", " << rotation_error << ", " << translation_error << ", " << reg_time_ms
                       << ", " << total_time_ms << std::endl;
            num_frames++;
        }
    }
    result_csv.close();
    return 0;
}

/*****************ExpKITTI end******************/

} // namespace demo
