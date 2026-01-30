/**
 * @file exp_3DMatch.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Entire pipeline for 3DMatch/3DLoMatch dataset evaluation
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
#include "utils_3DMatch.h"

#include "matcher.hpp"
#include "registration_utils.hpp"

#include <glog/logging.h>

#include <chrono>
#include <fstream>
#include <string>
#include <vector>

namespace demo {

template <typename FeatureSignature, class FeatureWizard> class Exp3DMatch : public ExpBase<FeatureSignature> {
  public:
    using FeatureCloud = typename ExpBase<FeatureSignature>::FeatureCloud;
    using FeatureCloudPtr = typename ExpBase<FeatureSignature>::FeatureCloudPtr;

    Exp3DMatch() = default;
    ~Exp3DMatch() = default;

    int evaluate(const std::string& data_path, const RealDataConfig& config) override;

    void parseDataSets(const RealDataConfig& config);

    void getSaliencyScore(const std::string& file_name, std::vector<float>& saliency_scores);

    void resamplingbyWeights(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, FeatureCloudPtr& cloud_feature,
                             std::vector<float>& weights, int num_sample = 5000);

  protected:
    static constexpr std::array<std::string_view, 8> m_folder_list = {
        "7-scenes-redkitchen",
        "sun3d-home_at-home_at_scan1_2013_jan_1",
        "sun3d-home_md-home_md_scan9_2012_sep_30",
        "sun3d-hotel_uc-scan3",
        "sun3d-hotel_umd-maryland_hotel1",
        "sun3d-hotel_umd-maryland_hotel3",
        "sun3d-mit_76_studyroom-76-1studyroom2",
        "sun3d-mit_lab_hj-lab_hj_tea_nov_2_2012_scan1_erika"};
    std::vector<RGBDTrajectory> m_trajectory;
    FeatureWizard m_feat_reader;
};

template <typename FeatureSignature>
using Exp3DMatchHandCraft = Exp3DMatch<FeatureSignature, FeatWizardHandCraft<FeatureSignature>>;

template <typename FeatureSignature>
using Exp3DMatchProcessed = Exp3DMatch<FeatureSignature, FeatWizardProcessed<FeatureSignature>>;

/*****************Implementation******************/

/*****************Exp3DMatch******************/

template <typename FeatureSignature, class FeatureWizard>
void Exp3DMatch<FeatureSignature, FeatureWizard>::parseDataSets(const RealDataConfig& config) {
    m_trajectory.clear();
    m_trajectory.reserve(m_folder_list.size());
    std::string gt_path;
    if (config.data_set == e3DMatch)
        gt_path = "../benchmarks/3DMatch/";
    else if (config.data_set == e3DLoMatch)
        gt_path = "../benchmarks/3DLoMatch/";

    for (auto& folder : m_folder_list) {
        RGBDTrajectory traj;
        std::string filename;
        filename.append(gt_path).append(folder).append("/gt.log");
        traj.LoadFromFile(filename);
        m_trajectory.push_back(traj);
    }
}

template <typename FeatureSignature, class FeatureWizard>
int Exp3DMatch<FeatureSignature, FeatureWizard>::evaluate(const std::string& data_path, const RealDataConfig& config) {
    int recall_sum = 0;
    int num_frames = 0;
    std::string data_type; // 3DMatch or 3DLoMatch
    if (config.data_set == e3DMatch)
        data_type = "3DMatch";
    else if (config.data_set == e3DLoMatch)
        data_type = "3DLoMatch";
    // Change it for long-term consecutive evaluation
    constexpr int start_frame = 0;
    std::string gt_path = "benchmarks/" + data_type;
    this->parseDataSets(config);
    std::ofstream result_csv("./result_" + data_type + ".csv", std::ios::out);
    result_csv << "Dataset, Source, Target, RE, TE, Time" << std::endl;
    gmor::FeatureMatcher<FeatureSignature, gmor::KdTreeFLANN<FeatureSignature>> matcher(config.knn, config.df);
    for (size_t i = 0; i < this->m_trajectory.size(); i++) { // Folder names
        std::string prefix;
        if (config.feature == ePredator) {
            prefix.append(data_path).append("/Predator/").append(data_type);
        } else {
            prefix.append(data_path).append("/").append(m_folder_list[i]).append("/cloud_bin_");
        }
        for (auto& [id1, data_vec] : this->m_trajectory[i].data_) {
            std::string src_prefix;
            pcl::PointCloud<pcl::PointXYZ>::Ptr src(new pcl::PointCloud<pcl::PointXYZ>);
            FeatureCloudPtr src_features(new FeatureCloud);
            if (config.feature == ePredator) {
            } else {
                src_prefix = prefix + std::to_string(id1);
                this->m_feat_reader.getPointCloudandFeatures(src_prefix, config, src, src_features);
            }

            for (auto& frame : data_vec) {
                if (num_frames < start_frame) {
                    num_frames++;
                    continue;
                }
                std::string tgt_prefix;
                pcl::PointCloud<pcl::PointXYZ>::Ptr tgt(new pcl::PointCloud<pcl::PointXYZ>);
                FeatureCloudPtr tgt_features(new FeatureCloud);
                std::vector<float> src_saliency_scores, tgt_saliency_scores;
                // Generate correspondences
                std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
                pcl::Correspondences corrs;
                if (config.feature == ePredator) {
                    src_prefix = prefix + "/tgt_" + std::to_string(num_frames);
                    tgt_prefix = prefix + "/src_" + std::to_string(num_frames);
                    this->m_feat_reader.getPointCloudandFeatures(src_prefix, config, src, src_features);
                    this->m_feat_reader.getPointCloudandFeatures(tgt_prefix, config, tgt, tgt_features);
                    std::string src_saliency_file_name = src_prefix + "_saliency_score.csv";
                    std::string tgt_saliency_file_name = tgt_prefix + "_saliency_score.csv";
                    getSaliencyScore(src_saliency_file_name, src_saliency_scores);
                    getSaliencyScore(tgt_saliency_file_name, tgt_saliency_scores);
                    resamplingbyWeights(src, src_features, src_saliency_scores, 5000);
                    resamplingbyWeights(tgt, tgt_features, tgt_saliency_scores, 5000);
                    matcher.matchwithSaliency(src_features, tgt_features, corrs, src_saliency_scores,
                                              tgt_saliency_scores, gmor::eCross | gmor::eSoftmax);
                } else {
                    tgt_prefix = prefix + std::to_string(frame.id2_);
                    this->m_feat_reader.getPointCloudandFeatures(tgt_prefix, config, tgt, tgt_features);
                    matcher.match(src_features, tgt_features, corrs, gmor::eCross | gmor::eSoftmax);
                }

                LOG(INFO) << m_folder_list[i] << " " << id1 << " to " << frame.id2_;
                LOG(INFO) << "src_feat: " << src_features->size() << ", tgt_feat: " << tgt_features->size()
                          << ", corrs: " << corrs.size();

                Eigen::Matrix4f transform_result;
                double reg_time_ms = demo::globalRegistrationPCL(src, tgt, corrs, config, transform_result);
                std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
                double total_time_ms =
                    std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(end - start).count() * 1e-3;
                LOG(INFO) << "Registration time: " << reg_time_ms << " ms, total time: " << total_time_ms << " ms";

                // Compare with ground truth
                Eigen::Matrix4d transform_gt = frame.transformation_;
                Eigen::Matrix4f transform_gt_float = transform_gt.inverse().cast<float>();

                auto rotation_error = gmor::getRotDist<float>(transform_result.topLeftCorner<3, 3>(),
                                                              transform_gt_float.topLeftCorner<3, 3>(), true);
                auto translation_error =
                    (transform_result.topRightCorner<3, 1>() - transform_gt_float.topRightCorner<3, 1>()).norm();
                if (rotation_error < 15 && translation_error < 0.3) {
                    recall_sum++;
                } else {
                    Eigen::AngleAxisf axis(transform_result.topLeftCorner<3, 3>());
                    Eigen::AngleAxisf axis_gt(transform_gt_float.topLeftCorner<3, 3>());
                    LOG(WARNING) << "Rotation axis error: " << (axis.axis() - axis_gt.axis()).norm();
                    LOG(WARNING) << "Translation offset: "
                                 << (transform_result.topRightCorner<3, 1>() -
                                     transform_gt_float.topRightCorner<3, 1>())
                                        .transpose();
                    LOG(WARNING) << "Translation along axis error: "
                                 << transform_result.topRightCorner<3, 1>().dot(axis_gt.axis());
                }
                num_frames++;
                LOG(INFO) << "RE: " << rotation_error << " (deg), " << "TE: " << translation_error
                          << ", RR: " << (recall_sum * 100.0) / (num_frames - start_frame) << "%";

                // Write to csv
                result_csv << m_folder_list[i] << ", " << id1 << ", " << frame.id2_ << ", " << rotation_error << ", "
                           << translation_error << ", " << reg_time_ms << std::endl;
            }
        }
    }
    result_csv.close();
    return 0;
}

template <typename FeatureSignature, class FeatureWizard>
void Exp3DMatch<FeatureSignature, FeatureWizard>::getSaliencyScore(const std::string& file_name,
                                                                   std::vector<float>& saliency_scores) {
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: failed to read file " << file_name << std::endl;
        return;
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val;
        std::getline(ss, val, ',');
        saliency_scores.push_back(std::stof(val));
    }
    file.close();
}

template <typename FeatureSignature, class FeatureWizard>
void Exp3DMatch<FeatureSignature, FeatureWizard>::resamplingbyWeights(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
                                                                      FeatureCloudPtr& cloud_feature,
                                                                      std::vector<float>& weights, int num_sample) {
    if (num_sample >= (int)cloud->size())
        return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
    FeatureCloudPtr cloud_feature_temp(new pcl::PointCloud<FeatureSignature>);
    std::vector<float> saliency_score_temp;

    std::vector<int> indices(weights.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](const auto& i, const auto& j) { return weights[i] > weights[j]; });

    for (int i = 0; i < num_sample; i++) {
        cloud_temp->push_back(cloud->points[indices[i]]);
        cloud_feature_temp->push_back(cloud_feature->points[indices[i]]);
        saliency_score_temp.push_back(weights[indices[i]]);
    };
    cloud = cloud_temp;
    cloud_feature = cloud_feature_temp;
    weights = std::move(saliency_score_temp);
}

/*****************Exp3DMatch end******************/

} // namespace demo
