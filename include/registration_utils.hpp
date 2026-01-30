/**
 * @file registration_utils.hpp
 * @author Zhao Zheng (zhengzhaobit@bit.edu.cn)
 * @brief Miscellaneous functions for registration.
 * @version 1.0.0
 * @date 2025-08-10
 *
 * @copyright Copyright (c) 2025. All rights reserved.
 * This source code is subject to the terms of the BSD-3-Clause license that can be found in the LICENSE file.
 *
 */

#pragma once

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include <random>
#include <stack>
#include <vector>

namespace gmor {

/**
 * @brief Distance between two rotation matrices.
 *
 * @tparam Scalar float or double.
 * @param R1 First rotation matrix.
 * @param R2 Second rotation matrix.
 * @param isDegree true in degrees, false in radians.
 * @return Scalar Rotation distance in radians or degrees.
 *
 * Typical calculation of distance between two rotation matrices is:
 * $$ \Delta \theta = \arccos(\frac{tr(R_2^T R_1)-1}{2}) $$
 * but it is not numerically stable when the angle is close to 0.
 * Recommended to be rewritten as:
 * $$ \Delta \theta = 2 \arcsin(\left \| R_1 - R_2 \right \|_F / 2 \sqrt{2}) $$
 * where $ \left \| \cdot \|_F $ is Frobenius norm.
 */
template <typename Scalar>
Scalar getRotDist(const Eigen::Matrix<Scalar, 3, 3>& R1, const Eigen::Matrix<Scalar, 3, 3>& R2, bool isDegree = true);

/**
 * @brief Get the Rotation Distance Quaternion.
 *
 * @tparam Scalar float or double.
 * @param q1 First quaternion.
 * @param q2 Second quaternion.
 * @param isDegree true in degrees, false in radians.
 * @return Scalar Rotation distance in radians or degrees.
 */
template <typename Scalar>
Scalar getRotDistQuat(const Eigen::Quaternion<Scalar>& q1, const Eigen::Quaternion<Scalar>& q2, bool isDegree = true);

/**
 * @brief Miller's projection in GIS.
 *
 * @param lon longitude in radians.
 * @param lat latitude in radians.
 * @return Eigen::Vector2d Miller projection coordinate.
 *
 * Miller's method in [12] John Parr Snyder. Map projections–A working manual, volume 1395. US Government
 * Printing Office, 1987. Latitude and longitude is in radians.
 */
template <typename Scalar = float> Eigen::Vector<Scalar, 2> millerProj(Scalar lon, Scalar lat);

/**
 * @brief Inverse Miller's projection in GIS.
 *
 * @param x Miller projection coordinate x.
 * @param y Miller projection coordinate y.
 * @return Eigen::Vector2d
 */
template <typename Scalar = float> Eigen::Vector<Scalar, 2> millerProjInv(Scalar x, Scalar y);

/**
 * @brief Transform Miller projection coordinate to rotation unit axis.
 *
 * @tparam Scalar float or double.
 * @param x Miller projection coordinate x.
 * @param y Miller projection coordinate y.
 * @return Eigen::Vector<Scalar, 3>
 */
template <typename Scalar = float> Eigen::Vector<Scalar, 3> miller2RotAxis(Scalar x, Scalar y);

/**
 * @brief Sample n elements of an N-element vector without replacement by shuffling. If n <= 3 or n >=
 * N, return the original vector.
 *
 * @tparam T Any type stored in std::vector.
 * @param v The vector to be sampled.
 * @param n Number of elements to be sampled.
 * @param seed_or_rd Use a given random seed (seed_or_rd >= 0) or std::random_device (seed_or_rd < 0).
 * @param halfTrick If true and n > v.size() / 2, only v.size() - n elements will be shuffled.
 * @return std::vector<T> Sampled vector.
 *
 * The original order of input vector will be preserved (Thread-unsafe). Time complexity is O(n), space complexity is
 * O(n).
 */
template <typename T> std::vector<T> sampleN(std::vector<T>& v, int n, int seed_or_rd = -1, bool halfTrick = false);

/**
 * @brief Generate two orthogonal vectors that are vertical to the given vector.
 *
 * @tparam Scalar float or double.
 * @param vec Input 3D vector.
 * @param v1 Vertical base vector on x-y plane.
 * @param v2 Vertical base vector.
 */
template <typename Scalar>
void genVertBaseVecs(const Eigen::Vector<Scalar, 3>& vec, Eigen::Vector<Scalar, 3>& v1, Eigen::Vector<Scalar, 3>& v2);

/**
 * @brief Fit plane points using eigen value decomposition (EVD).
 *
 * @tparam Scalar float or double.
 * @param points Input points.
 * @param normal Normal of the plane.
 * @param centroid Centroid of the points.
 * @return Scalar Squared error of the fitted plane.
 *
 */
template <typename Scalar>
Scalar planeFitting(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& points, Eigen::Vector<Scalar, 3>& normal,
                    Eigen::Vector<Scalar, 3>& centroid);

/*****************Implementation******************/

template <typename Scalar>
Scalar getRotDist(const Eigen::Matrix<Scalar, 3, 3>& R1, const Eigen::Matrix<Scalar, 3, 3>& R2, bool isDegree) {
    // double tmp = 0.5 * ((R1.transpose() * R2).trace() - 1);
    // tmp = std::min(std::max(tmp, -1.0), 1.0);
    // tmp = std::acos(tmp);
    double tmp = 0.25 * std::sqrt(2 * (R1 - R2).array().square().sum());
    tmp = std::min(tmp, 1.0);
    tmp = 2 * std::asin(tmp);
    return isDegree ? tmp * 180 * M_1_PI : tmp;
}

template <typename Scalar>
Scalar getRotDistQuat(const Eigen::Quaternion<Scalar>& q1, const Eigen::Quaternion<Scalar>& q2, bool isDegree) {
    double tmp = 0.5 * std::min((q1.coeffs() + q2.coeffs()).norm(), (q1.coeffs() - q2.coeffs()).norm());
    tmp = std::min(tmp, 1.0);
    tmp = 4 * std::asin(tmp);
    return isDegree ? tmp * 180 * M_1_PI : tmp;
}

template <typename Scalar> Eigen::Vector<Scalar, 2> millerProj(Scalar lon, Scalar lat) {
    // Eq. (11-2) in the manual
    Scalar y = 1.25 * log(tan(0.25 * M_PI + 0.4 * lat));
    Eigen::Vector<Scalar, 2> xy(lon, y);
    return xy;
}

template <typename Scalar> Eigen::Vector<Scalar, 2> millerProjInv(Scalar x, Scalar y) {
    // Eq. (11-6) in the manual
    Scalar lat = (atan(exp(0.8 * y)) * 2.5 - 0.625 * M_PI);
    Eigen::Vector<Scalar, 2> lonlat(x, lat);
    return lonlat;
}

template <typename Scalar> Eigen::Vector<Scalar, 3> miller2RotAxis(Scalar x, Scalar y) {
    Eigen ::Vector<Scalar, 2> lonlat = millerProjInv(x, y);
    Scalar lon = lonlat(0);
    Scalar lat = lonlat(1); // lat in [0, pi/2]
    Scalar x_rot = cos(lon) * cos(lat);
    Scalar y_rot = sin(lon) * cos(lat);
    Scalar z_rot = sin(lat);
    Eigen::Vector<Scalar, 3> rotation_axis(x_rot, y_rot, z_rot);
    return rotation_axis;
}

template <typename T> std::vector<T> sampleN(std::vector<T>& v, int n, int seed_or_rd, bool halfTrick) {
    if (n >= (int)v.size() || n <= 3)
        return v;
    unsigned int seed;
    if (seed_or_rd < 0) {
        std::random_device rd;
        seed = rd();
    } else {
        seed = seed_or_rd;
    }
    std::mt19937 g(seed);
    std::vector<T> v_sample;

    v_sample.resize(n);
    std::uniform_int_distribution<int> d(0, v.size() - 1);
    // Do not change the original vector
    std::stack<int> v_order;
    if (halfTrick && (n << 1) > (int)v.size()) {
        for (int i = 0; i < (int)v.size() - n; ++i) {
            int j = d(g);
            std::swap(v[i], v[j]);
            v_order.push(j);
        }
        std::copy(v.end() - n, v.end(), v_sample.begin());
        for (int i = v.size() - n - 1; i >= 0; --i) {
            int j = v_order.top();
            v_order.pop();
            std::swap(v[i], v[j]);
        }
    } else {
        for (int i = 0; i < n; ++i) {
            int j = d(g);
            std::swap(v[i], v[j]);
            v_order.push(j);
        }
        std::copy(v.begin(), v.begin() + n, v_sample.begin());
        for (int i = n - 1; i >= 0; --i) {
            int j = v_order.top();
            v_order.pop();
            std::swap(v[i], v[j]);
        }
    }

    return v_sample;
}

template <typename Scalar>
void genVertBaseVecs(const Eigen::Vector<Scalar, 3>& vec, Eigen::Vector<Scalar, 3>& v1, Eigen::Vector<Scalar, 3>& v2) {
    // vec is along z-axis
    if (std::abs(vec(0)) + std::abs(vec(1)) < 1e-6) {
        if (vec(2) > 0) {
            v1 << 1, 0, 0;
            v2 << 0, 1, 0;
        } else {
            v1 << -1, 0, 0;
            v2 << 0, 1, 0;
        }
        return;
    }
    // vec is not along z-axis, $ vec = (sin(\psi)cos(\theta), sin(\psi)sin(\theta), cos(\psi)) $
    Scalar psi = std::acos(vec(2));
    Scalar theta = std::atan2(vec(1), vec(0));
    // v1 is on x-y plane
    v1 << -std::sin(theta), std::cos(theta), 0;
    // v2 = vec.cross(v1);
    v2 << -std::cos(psi) * std::cos(theta), -std::cos(psi) * std::sin(theta), std::sin(psi);
}

template <typename Scalar>
Scalar planeFitting(const Eigen::Matrix<Scalar, 3, Eigen::Dynamic>& points, Eigen::Vector<Scalar, 3>& normal,
                    Eigen::Vector<Scalar, 3>& centroid) {
    centroid = points.rowwise().mean();
    Eigen::Matrix<Scalar, 3, 3> cov = points * points.transpose() - points.cols() * centroid * centroid.transpose();
    // Default ascending order
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<Scalar, 3, 3>> eigensolver(cov);
    normal = eigensolver.eigenvectors().col(0);
    return eigensolver.eigenvalues()(0) / points.cols();
}

} // namespace gmor
