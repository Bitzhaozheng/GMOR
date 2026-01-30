/**
 * @file exp_io.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Read points and features from file.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>

#include <Eigen/Core>

#include <iostream>
#include <string>

namespace demo {

// Load point cloud from file (*.ply, *.pcd, *.bin or *.csv)
template <typename PointT>
int readPointCloud(const std::string& file_name, typename pcl::PointCloud<PointT>::Ptr& cloud);

// Read feature cloud from csv file
template <typename FeatureSignature>
int readFeatureCloud(const std::string& file_name, typename pcl::PointCloud<FeatureSignature>::Ptr& cloud,
                     bool normalize = false);
template <typename FeatureSignature> void getFeature(const std::string& line, FeatureSignature& feature);

/*****************Implementation******************/

template <typename PointT>
int readPointCloud(const std::string& file_name, typename pcl::PointCloud<PointT>::Ptr& cloud) {
    cloud.reset(new pcl::PointCloud<PointT>);
    std::string ext = file_name.substr(file_name.find_last_of('.') + 1);

    if (ext == "ply") {
        if (pcl::io::loadPLYFile(file_name, *cloud) < 0) {
            std::cerr << "Error: failed to read file " << file_name << std::endl;
            return -1;
        }
    } else if (ext == "pcd") {
        if (pcl::io::loadPCDFile(file_name, *cloud) < 0) {
            std::cerr << "Error: failed to read file " << file_name << std::endl;
            return -1;
        }
    } else if (ext == "bin") { // LiDAR point cloud (float32, x, y, z, intensity)
        std::ifstream file(file_name, std::ios::in | std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Error: failed to read file " << file_name << std::endl;
            return -1;
        }
        while (!file.eof()) {
            PointT pt;                                                      // PointXYZI, PointXYZ (16 bytes aligned)
            file.read(reinterpret_cast<char*>(pt.data), 4 * sizeof(float)); // x, y, z, intensity
            cloud->push_back(pt);
        }
        file.close();
    } else if (ext == "csv") {
        std::ifstream file(file_name);
        if (!file.is_open()) {
            std::cerr << "Error: failed to read file " << file_name << std::endl;
            return -1;
        }
        std::string line;
        while (std::getline(file, line)) {
            PointT pt;
            std::stringstream ss(line);
            std::string val;

            for (int i = 0; i < 3; ++i) {
                std::getline(ss, val, ',');
                pt.data[i] = std::stof(val);
            }
            cloud->push_back(pt);
        }
        file.close();
    }
    return 0;
}

template <typename FeatureSignature>
int readFeatureCloud(const std::string& file_name, typename pcl::PointCloud<FeatureSignature>::Ptr& cloud,
                     bool normalize) {
    cloud.reset(new pcl::PointCloud<FeatureSignature>);
    std::ifstream file(file_name);
    if (!file.is_open()) {
        std::cerr << "Error: failed to read file " << file_name << std::endl;
        return -1;
    }
    std::string line;
    while (std::getline(file, line)) {
        FeatureSignature feature;
        getFeature(line, feature);
        if (normalize)
            Eigen::Map<Eigen::Vector<float, FeatureSignature::descriptorSize()>>(feature.histogram).normalize();
        cloud->push_back(feature);
    }
    file.close();
    return 0;
}

template <typename FeatureSignature> void getFeature(const std::string& line, FeatureSignature& feature) {
    std::stringstream ss(line);
    std::string val;
    int i = 0;
    while (std::getline(ss, val, ',')) {
        if (i >= FeatureSignature::descriptorSize()) {
            std::cerr << "Warning: feature size is larger than " << FeatureSignature::descriptorSize() << std::endl;
            break;
        }
        feature.histogram[i] = std::stof(val);
        ++i;
    }
    if (i < FeatureSignature::descriptorSize()) {
        std::cerr << "Warning: feature size is less than " << FeatureSignature::descriptorSize() << std::endl;
    }
}

} // namespace demo
