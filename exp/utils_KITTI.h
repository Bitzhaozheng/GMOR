/**
 * @file utils_KITTI.h
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Parsing KITTI odometry with benchmark pairs.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <fstream>
#include <iostream>
#include <vector>

struct FrameKITTI {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    int id1_, id2_;
    Eigen::Matrix4d transformation_;
    FrameKITTI(int id1, int id2, Eigen::Matrix4d& t) : id1_(id1), id2_(id2), transformation_(t) {}
};

struct TrajectoryKITTI {
    std::vector<FrameKITTI, Eigen::aligned_allocator<FrameKITTI>> data_;

    void LoadFromFile(const std::string& pair_name, const std::string& poses_name, const std::string& calib_name) {
        data_.clear();
        Eigen::Matrix4d trans;

        std::cout << "Loading trajectory from " << pair_name << " and " << poses_name << std::endl;
        // Read calibration matrix
        std::ifstream fs(calib_name);
        if (!fs.is_open()) {
            std::cerr << "Error: cannot open file " << calib_name << std::endl;
            return;
        }
        std::string line;
        Eigen::Isometry3d velo_to_camera;
        Eigen::Matrix4d calib_mat;
        while (std::getline(fs, line)) {
            std::stringstream ss(line);
            std::string str;
            std::getline(ss, str, ' ');
            if (str == "Tr:") {
                for (int i = 0; i < 12; i++) {
                    std::getline(ss, str, ' ');
                    calib_mat(i / 4, i % 4) = std::stod(str);
                }
                calib_mat(3, 0) = calib_mat(3, 1) = calib_mat(3, 2) = 0;
                calib_mat(3, 3) = 1;
                velo_to_camera = Eigen::Isometry3d(calib_mat);
                break;
            } else {
                continue;
            }
        }
        fs.close();

        // Read benchmark pairs
        fs.open(pair_name);
        if (!fs.is_open()) {
            std::cerr << "Error: cannot open file " << pair_name << std::endl;
            return;
        }
        std::vector<std::pair<int, int>> pairs;
        while (std::getline(fs, line)) {
            std::stringstream ss(line);
            std::string str1, str2;
            std::getline(ss, str1, ',');
            std::getline(ss, str2);
            pairs.emplace_back(std::stoi(str1), std::stoi(str2));
        }
        fs.close();

        // Read poses by pairs
        fs.open(poses_name);
        if (!fs.is_open()) {
            std::cerr << "Error: cannot open file " << poses_name << std::endl;
            return;
        }

        int count = 0;
        int id_last = -1;
        // Ascending order of pairs: pairs[i].id1 < pairs[i].id2 < pairs[i+1].id1 < pairs[i+1].id2
        for (const auto& pair : pairs) {
            auto id1 = pair.first;
            auto id2 = pair.second; // id1 < id2
            if (id_last >= id1 || id1 >= id2) {
                std::cerr << "Warning: invalid order id_last: " << id_last << ", id1: " << id1 << ", id2: " << id2
                          << std::endl;
                continue;
            }
            Eigen::Isometry3d pose1, pose2;
            while (std::getline(fs, line)) {
                if (count == id1) {
                    Eigen::Matrix4d pose_mat;
                    std::stringstream ss(line);
                    std::string str;
                    for (int i = 0; i < 12; i++) {
                        std::getline(ss, str, ' ');
                        pose_mat(i / 4, i % 4) = std::stod(str);
                    }
                    pose_mat(3, 0) = pose_mat(3, 1) = pose_mat(3, 2) = 0;
                    pose_mat(3, 3) = 1;
                    pose1 = Eigen::Isometry3d(pose_mat);
                }

                if (count == id2) {
                    Eigen::Matrix4d pose_mat;
                    std::stringstream ss(line);
                    std::string str;
                    for (int i = 0; i < 12; i++) {
                        std::getline(ss, str, ' ');
                        pose_mat(i / 4, i % 4) = std::stod(str);
                    }
                    pose_mat(3, 0) = pose_mat(3, 1) = pose_mat(3, 2) = 0;
                    pose_mat(3, 3) = 1;
                    pose2 = Eigen::Isometry3d(pose_mat);
                    id_last = id2;
                    count++;
                    break;
                }
                count++;
            }

            // Transformation matrix from velodyne 1 to 2
            trans = (velo_to_camera.inverse() * pose2.inverse() * pose1 * velo_to_camera).matrix();
            data_.emplace_back(id1, id2, trans);
        }
        fs.close();

        // 306 + 162 + 87 = 555 pairs in general
        std::cout << "Trajectory loaded. Size: " << data_.size() << std::endl;
    }
};
